import torch
from collections import namedtuple
from typing import Optional
from datasets import load_dataset, Features, Sequence, Value

# token_id, depth, parent_idx, children (list of child flat indices in BFS order)
Node = namedtuple("Node", ["token_id", "depth", "parent_idx", "children"])

def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [(num_target_layers // 2)]
    start = 1
    end = num_target_layers - 3
    span = end - start
    target_layer_ids = [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]
    return target_layer_ids

def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
) -> torch.Tensor:
    offset = 1
    selected_states = []
    for layer_id in layer_ids:
        selected_states.append(hidden_states[layer_id + offset])
    target_hidden = torch.cat(selected_states, dim=-1)
    return target_hidden

def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)

def sample_tree(logits: torch.Tensor, max_num_nodes: int = 2) -> torch.Tensor:
    """Sample top-k token indices for each position. Returns [bsz, seq_len, max_num_nodes]."""
    _, indices = torch.topk(logits, k=max_num_nodes, dim=-1)
    return indices


def build_verify_tree(
    tree_config: torch.Tensor,
    all_drafts: torch.Tensor,
    start: int,
    root_token: torch.Tensor,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Build the tree structure for speculative verification.

    Args:
        tree_config: [bsz, num_depths] - 每层分支数，v 表示 v 个节点。0 表示无分支且后续层都不创建
        all_drafts: [bsz, num_depths, max_num_nodes] - token IDs for each position
        start: cache length / start position for absolute position IDs
        root_token: [bsz, 1] - root token to prepend (required for pred-after-root verification)
        pad_token_id: token ID used for padding when batch items have different tree sizes

    Returns:
        tree_ids: [bsz, total_tokens] - flattened token IDs (with root if provided)
        tree_position_ids: [bsz, total_tokens] - position IDs for RoPE
        tree_attention_mask: [bsz, 1, total_tokens, start+total_tokens] - tree attention mask
        tree_info: dict with 'all_nodes', 'nodes_at_depth' per batch for acceptance
    """
    bsz, num_depths, max_num_nodes = all_drafts.shape
    device = all_drafts.device

    # tree_config[i] = 该层分支数，0 表示无分支且后续层都不创建
    num_branches = tree_config.clamp(0, max_num_nodes)

    all_nodes: list[list] = []
    all_nodes_at_depth: list[list[int]] = []

    for b in range(bsz):
        n0 = num_branches[b, 0].item()
        if n0 == 0:
            all_nodes.append([])
            all_nodes_at_depth.append([0])
            continue

        nodes: list = []
        nodes_at_depth = [n0]
        for i in range(n0):
            nodes.append(Node(all_drafts[b, 0, i], 0, -1, []))

        for d in range(1, num_depths):
            n_d = num_branches[b, d].item()
            if n_d == 0:
                break
            num_parents = nodes_at_depth[-1]
            parent_base = sum(nodes_at_depth[:-1]) if d > 1 else 0

            for p in range(num_parents):
                parent_idx = parent_base + p
                for c in range(n_d):
                    nodes.append(Node(all_drafts[b, d, c], d, parent_idx, []))
            nodes_at_depth.append(num_parents * n_d)

        for d in range(1, len(nodes_at_depth)):
            n_d = num_branches[b, d].item()
            num_parents = nodes_at_depth[d - 1]
            base = sum(nodes_at_depth[:d])
            for p in range(num_parents):
                parent_idx = sum(nodes_at_depth[: d - 1]) + p
                for c in range(n_d):
                    child_idx = base + p * n_d + c
                    nodes[parent_idx].children.append(child_idx)

        all_nodes.append(nodes)
        all_nodes_at_depth.append(nodes_at_depth)

    max_len = max(len(nodes) for nodes in all_nodes)

    total_len = 1 + max_len
    tree_ids = torch.full((bsz, total_len), pad_token_id, dtype=torch.long, device=device)
    tree_ids[:, 0] = root_token.squeeze(-1)  # root_token: [bsz, 1] -> [bsz]
    tree_position_ids = torch.zeros(bsz, total_len, dtype=torch.long, device=device)
    tree_position_ids[:, 0] = start  # root is first new token (cache has length start)
    tree_attention_mask = torch.full(
        (bsz, 1, total_len, start + total_len),
        float("-inf"),
        device=device,
        dtype=torch.float32,
    )
    tree_offset = 1  # logits[0]=pred after root, logits[1+i]=pred after tree node i

    for b in range(bsz):
        nodes = all_nodes[b]
        L = len(nodes)
        if L > 0:
            tree_ids[b, tree_offset : tree_offset + L] = torch.stack([n.token_id for n in nodes])
            tree_position_ids[b, tree_offset : tree_offset + L] = torch.tensor(
                [start + 1 + n.depth for n in nodes], dtype=torch.long, device=device
            )
        tree_attention_mask[b, 0, 0, :start] = 0.0  # root attends to cache
        tree_attention_mask[b, 0, 0, start] = 0.0  # root attends to self
        for i in range(L):
            pos = tree_offset + i
            tree_attention_mask[b, 0, pos, :start] = 0.0  # attend to cache
            tree_attention_mask[b, 0, pos, start] = 0.0  # attend to root
            idx = i
            while idx >= 0:
                tree_attention_mask[b, 0, pos, start + tree_offset + idx] = 0.0
                idx = nodes[idx].parent_idx

    tree_info = {
        "all_nodes": all_nodes,
        "nodes_at_depth": all_nodes_at_depth,
        "tree_offset": tree_offset,
    }
    return tree_ids, tree_position_ids, tree_attention_mask, tree_info


def compute_tree_acceptance(
    tree_info: dict,
    tree_ids: torch.Tensor,
    all_drafts: torch.Tensor,
    logits: torch.Tensor,
    temperature: float = 0.0,
) -> tuple[int, torch.Tensor, torch.Tensor, list[int]]:
    """
    Find the longest accepting path in the tree by checking all branches.

    Args:
        tree_info: from build_verify_tree
        tree_ids: [bsz, seq_len] - token IDs (tree_ids[tree_offset+i] = token at tree node i)
        all_drafts: [bsz, num_depths, max_num_nodes]
        logits: [bsz, seq_len, vocab] - logits[0]=pred after root, logits[tree_offset+i]=pred after node i
        temperature: for sampling next token

    Returns:
        acceptance_length: number of draft tokens accepted
        accepted_tokens: [bsz, acceptance_length] the accepted path (for updating output_ids)
        next_token: [bsz] the token to append after accepted path
        path_indices: list of tree node indices (for b=0) for extracting target_hidden
    """
    bsz = logits.shape[0]
    device = logits.device
    tree_offset = tree_info["tree_offset"]
    all_nodes = tree_info["all_nodes"]
    nodes_at_depth = tree_info["nodes_at_depth"]

    if temperature < 1e-5:
        posterior = logits.argmax(dim=-1)
    else:
        posterior = torch.multinomial(
            torch.softmax(logits / temperature, dim=-1).view(-1, logits.shape[-1]),
            num_samples=1,
        ).view(bsz, logits.shape[1])

    best_lengths = []
    best_paths = []
    best_path_indices = []
    best_nexts = []

    for b in range(bsz):
        best_len = 0
        best_path = []
        best_path_idx = []
        best_next = None

        pred_after_root = posterior[b, 0]
        n0 = nodes_at_depth[b][0]
        n_depths = len(nodes_at_depth[b])

        def dfs(node_idx: int, depth: int, path: list, path_idx: list) -> None:
            nonlocal best_len, best_path, best_path_idx, best_next
            node = all_nodes[b][node_idx]
            pred = posterior[b, tree_offset + node_idx]
            if depth == n_depths - 1:
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path.copy()
                    best_path_idx = path_idx.copy()
                    best_next = pred
                return
            for child_idx in node.children:
                child_token = tree_ids[b, tree_offset + child_idx]
                if child_token.item() == pred.item():
                    dfs(child_idx, depth + 1, path + [child_token], path_idx + [child_idx])
            if len(path) > best_len:
                best_len = len(path)
                best_path = path.copy()
                best_path_idx = path_idx.copy()
                best_next = pred

        for i in range(n0):
            if all_drafts[b, 0, i].item() == pred_after_root.item():
                dfs(i, 0, [all_drafts[b, 0, i]], [i])

        if best_len == 0:
            best_next = pred_after_root

        best_lengths.append(best_len)
        best_paths.append(best_path)
        best_path_indices.append(best_path_idx)
        best_nexts.append(best_next)

    max_len = max(best_lengths) if best_lengths else 0
    accepted_tokens = torch.zeros(bsz, max(1, max_len), dtype=torch.long, device=device)
    for b in range(bsz):
        if best_paths[b]:
            accepted_tokens[b, : len(best_paths[b])] = torch.stack(best_paths[b])
    next_tokens = torch.stack(
        [n.unsqueeze(0) if n.dim() == 0 else n for n in best_nexts]
    ).squeeze(-1).to(device)

    acceptance_length = best_lengths[0] if bsz == 1 else min(best_lengths)
    path_indices = best_path_indices[0] if bsz > 0 else []
    return acceptance_length, accepted_tokens, next_tokens, path_indices

def load_and_process_dataset(data_name: str):
    # Math datasets
    if data_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        prompt_fmt = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
    
    elif data_name == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
    
    elif data_name == "aime24":
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "aime25":
        dataset = load_dataset("MathArena/aime_2025", split="train")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    # Chat datasets 
    elif data_name == "alpaca":
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        dataset = dataset.map(lambda x: {"formatted_input": (f"{x['instruction']}\n\nInput:\n{x['input']}" if x['input'] else x['instruction'])})
        dataset = dataset.map(lambda x: {"turns": [x["formatted_input"]]})

    elif data_name == "mt-bench":
        dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        dataset = dataset.map(lambda x: {"turns": x["prompt"]})

    # Coding datasets
    elif data_name == "humaneval":
        dataset = load_dataset("openai/openai_humaneval", split="test")
        prompt_fmt = "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{prompt}\n```"
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
    
    elif data_name == "mbpp":
        dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        dataset = dataset.map(lambda x: {"turns": [x["prompt"]]})
    
    elif data_name == "lbpp":
        LBPP_PY_TEST_URL = "https://huggingface.co/datasets/CohereLabs/lbpp/resolve/main/python/test.parquet"
        dataset = load_dataset("parquet", data_files={"test": LBPP_PY_TEST_URL})["test"]
        dataset = dataset.map(lambda x: {"turns": [x["instruction"]]})

    elif data_name == "swe-bench":
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        prompt_fmt = "Problem Statement:\n{problem_statement}\nPlease fix the issue described above."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
    
    elif data_name == "livecodebench":
        base = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/"
        allowed_files = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]
        urls = [base + fn for fn in allowed_files]
        dataset = load_dataset("json", data_files={"test": urls})["test"]
        def format_lcb(doc):
            system_prompt = (
                "You are an expert Python programmer. You will be given a question (problem specification) "
                "and will generate a correct Python program that matches the specification and passes all tests. "
                "You will NOT return anything except for the program"
            )
            question_block = f"### Question:\n{doc['question_content']}"
            if doc.get("starter_code"):
                format_message = "### Format: Use the following code structure:"
                code_block = f"```python\n{doc['starter_code']}\n```"
            else:
                format_message = "### Format: Write your code in the following format:"
                code_block = "```python\n# YOUR CODE HERE\n```"
            answer_footer = "### Answer: (use the provided format with backticks)"
            return f"{system_prompt}\n\n{question_block}\n\n{format_message}\n{code_block}\n\n{answer_footer}"
        target_features = Features({"turns": Sequence(Value("large_string"))})
        dataset = dataset.map(
            lambda x: {"turns": [format_lcb(x)]},
            remove_columns=dataset.column_names,
            features=target_features
        )
    
    return dataset