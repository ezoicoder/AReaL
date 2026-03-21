import random
from dataclasses import dataclass, field
from math import ceil


def _get_stats(
    lens: list[int], lcp_lens: list[int], mode: str, block_size: int | None = None
) -> dict:
    n_tree_tokens = sum(lens) - sum(lcp_lens)
    sum_depth = 0
    for i in range(len(lens)):
        start = lcp_lens[i - 1] if i > 0 else 0
        end = lens[i]
        sum_depth += (start + end - 1) * (end - start) // 2

    if mode == "forward":
        sum_prefix_len = sum(lcp_lens)

        return {
            "n_leaf_sequences": len(lens),
            "n_tree_tokens": n_tree_tokens,
            "sum_prefix_len": sum_prefix_len,
            "sum_depth": sum_depth,
        }

    elif mode == "backward":
        sum_prefix_len = 0
        n_f1_tokens = 0
        for i in range(len(lens)):
            start = lcp_lens[i] if i < len(lcp_lens) else 0
            end = lens[i]
            pop_len = end - start
            f1_start = lcp_lens[i - 1] if i > 0 else 0

            if block_size is None or pop_len <= block_size:
                f1_end = lcp_lens[i]
                sum_prefix_len += start
            else:
                n_blocks = ceil(pop_len / block_size)
                block_size_actual = ceil(pop_len / n_blocks)
                f1_end = end - block_size_actual
                for b in range(n_blocks):
                    pop_start = max(end - (b + 1) * block_size_actual, start)
                    sum_prefix_len += pop_start

            n_f1_tokens += max(f1_end - f1_start, 0)

        return {
            "n_leaf_sequences": len(lens),
            "n_tree_tokens": n_tree_tokens,
            "sum_prefix_len": sum_prefix_len,
            "sum_depth": sum_depth,
            "n_f1_tokens": n_f1_tokens,
        }

    else:
        raise ValueError(f"Unsupported mode: {mode}")


@dataclass(slots=True)
class CTNode:
    """压缩Trie树节点"""

    depth: int = 0  # 节点深度
    seq_id: int = -1  # 字符串编号，-1 表示内部节点
    chain_tail_depth: int = 0  # 主链末端的深度
    child_ids: list[int] = field(default_factory=list)  # 子节点ID列表


class CompressedTrie:
    """压缩Trie树类，用于规划遍历顺序"""

    def __init__(self, lens: list[int], lcp_lens: list[int]):
        """
        初始化压缩Trie树

        Args:
            lens: 每个字符串的长度，保证按字典序排序
            lcp_lens: 相邻字符串的LCP长度，len(lcp_lens) = len(lens) - 1
        """
        if len(lcp_lens) != len(lens) - 1:
            raise ValueError("lcp_lens的长度必须为len(lens)-1")

        self.nodes: list[CTNode] = []  # 存储所有节点
        self._build(lens, lcp_lens)

        self.lca_depth = None
        self.order = None
        self.lens = None
        self.lcp_lens = None

    def _new_node(self, depth: int, seq_id: int = -1) -> int:
        """创建新节点，返回节点ID"""
        self.nodes.append(CTNode(depth=depth, seq_id=seq_id))
        return len(self.nodes) - 1

    def _build(self, lens: list[int], lcp_lens: list[int]):
        """构建压缩Trie树"""

        n_seqs = len(lens)
        # 创建根节点
        root_id = self._new_node(depth=0, seq_id=-1)
        stack = [(root_id, 0)]  # 栈结构：(node_id, depth)
        nodes = self.nodes

        for seq_id in range(n_seqs):
            len_i = lens[seq_id]
            lcp = lcp_lens[seq_id - 1] if seq_id > 0 else 0

            if len(stack) >= 2:
                while stack[-2][1] > lcp:
                    # 弹出子节点，并将子节点连接到父节点
                    child_id = stack.pop()[0]
                    parent_id = stack[-1][0]
                    nodes[parent_id].child_ids.append(child_id)

                child_id = stack.pop()[0]
                if stack[-1][1] < lcp:
                    lcp_node_id = self._new_node(depth=lcp, seq_id=-1)
                    stack.append((lcp_node_id, lcp))
                parent_id = stack[-1][0]
                nodes[parent_id].child_ids.append(child_id)
            else:
                if stack[-1][1] < lcp:
                    lcp_node_id = self._new_node(depth=lcp, seq_id=-1)
                    stack.append((lcp_node_id, lcp))

            # 创建新的叶节点
            parent_id = stack[-1][0]
            cur_node_id = self._new_node(depth=len_i, seq_id=seq_id)
            stack.append((cur_node_id, len_i))

        while len(stack) >= 2:
            child_id = stack.pop()[0]
            parent_id = stack[-1][0]
            nodes[parent_id].child_ids.append(child_id)

    def dfs_chain(self, node_id: int, child_order_func) -> int:
        """计算每个节点的 chain_tail_depth"""
        node = self.nodes[node_id]

        # 如果是叶节点
        if node.seq_id != -1:
            node.chain_tail_depth = node.depth
            return

        for child_id in node.child_ids:
            self.dfs_chain(child_id, child_order_func)

        child_ids = child_order_func(node_id)
        node.chain_tail_depth = self.nodes[child_ids[0]].chain_tail_depth

    def dfs_get_lens(self, node_id: int, seq_set: set[int]):
        node = self.nodes[node_id]

        if node.seq_id != -1:
            if node.seq_id in seq_set:
                self.lens.append(node.depth)
                self.lcp_lens.append(self.lca_depth)
                self.lca_depth = node.depth
            return

        for child_id in node.child_ids:
            self.lca_depth = min(self.lca_depth, node.depth)
            self.dfs_get_lens(child_id, seq_set)

    def get_lens(self, seq_set: set[int]):
        self.lens = []
        self.lcp_lens = []
        self.lca_depth = 0
        self.dfs_get_lens(0, seq_set)
        return self.lens, self.lcp_lens[1:]

    def dfs_get_order(self, node_id: int, child_order_func):
        node = self.nodes[node_id]

        # 如果是叶节点，记录字符串编号
        if node.seq_id != -1:
            self.order.append(node.seq_id)
            self.lens.append(node.depth)
            self.lcp_lens.append(self.lca_depth)
            self.lca_depth = node.depth
            return

        # 根据指定的顺序函数获取子节点遍历顺序
        child_ids = child_order_func(node_id)

        # 递归遍历子节点
        for child_id in child_ids:
            self.lca_depth = min(self.lca_depth, node.depth)
            self.dfs_get_order(child_id, child_order_func)

    def _get_child_order_forward(self, node_id: int) -> list[int]:
        node = self.nodes[node_id]
        return sorted(
            node.child_ids, key=lambda child_id: self.nodes[child_id].chain_tail_depth
        )

    def _get_child_order_backward(self, node_id: int) -> list[int]:
        node = self.nodes[node_id]
        return sorted(
            node.child_ids,
            key=lambda child_id: (
                1 if self.nodes[child_id].child_ids else 0,
                self.nodes[child_id].chain_tail_depth,
            ),
        )

    def _get_child_order_random(
        self, node_id: int, seed: int | None = None
    ) -> list[int]:
        node = self.nodes[node_id]
        child_ids = node.child_ids.copy()

        if seed is not None:
            local_random = random.Random(seed)
            local_random.shuffle(child_ids)
        else:
            random.shuffle(child_ids)

        return child_ids

    def get_order(self, child_order_func):
        """获取按指定顺序函数DFS遍历得到的字符串顺序"""
        self.dfs_chain(0, child_order_func)
        self.order = []
        self.lens = []
        self.lcp_lens = []
        self.lca_depth = 0
        self.dfs_get_order(0, child_order_func)

    def get_order_forward(self):
        """获取按main_Ld优先DFS遍历得到的字符串顺序"""
        self.get_order(self._get_child_order_forward)
        return self.order, self.lens, self.lcp_lens[1:]

    def get_order_backward(self):
        """获取按main_Ld优先DFS遍历得到的字符串顺序"""
        self.get_order(self._get_child_order_backward)
        return self.order[::-1], self.lens[::-1], self.lcp_lens[1:][::-1]

    def get_order_random(self, seed: int | None = None):
        """获取随机打乱边表后的DFS遍历顺序"""
        self.get_order(lambda node_id: self._get_child_order_random(node_id, seed))
        return self.order


def _get_subtrie(trie, seq_set: set[int]) -> CompressedTrie:
    lens, lcp_lens = trie.get_lens(seq_set)
    return CompressedTrie(lens, lcp_lens)


# -------- Test --------


def test_compressed_trie():
    lens1 = [5, 4, 3, 2]
    lcp_lens1 = [3, 2, 1]

    trie1 = CompressedTrie(lens1, lcp_lens1)

    order, lens, lcp_lens = trie1.get_order_forward()
    print(order, lens, lcp_lens)

    order, lens, lcp_lens = trie1.get_order_backward()
    print(order, lens, lcp_lens)

    order, lens, lcp_lens = trie1.get_order_random()
    print(order, lens, lcp_lens)


if __name__ == "__main__":
    test_compressed_trie()
