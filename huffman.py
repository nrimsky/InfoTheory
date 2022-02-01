from dataclasses import dataclass, field
from collections import Counter
import heapq
from typing import Any


@dataclass
class Node:
    symbol: str
    frequency: float
    left = None
    right = None
    code = ""


@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any = field(compare=False)


def get_codes(root_node: Node):
    codes = {}

    def _calc_codes(node: Node, val):
        new_val = val + node.code
        if node.left:
            _calc_codes(node.left, new_val)
        if node.right:
            _calc_codes(node.right, new_val)
        if (not node.left) and (not node.right):
            codes[node.symbol] = new_val

    _calc_codes(root_node, "")
    return codes


def encode(data):
    counts = Counter(data)
    nodes = [PrioritizedItem(count, Node(symbol, count)) for symbol, count in counts.items()]
    heapq.heapify(nodes)
    while len(nodes) > 1:
        left = heapq.heappop(nodes).item
        right = heapq.heappop(nodes).item
        left.code = "0"
        right.code = "1"
        new_node = Node(left.symbol + right.symbol, left.frequency + right.frequency)
        new_node.left = left
        new_node.right = right
        heapq.heappush(nodes, PrioritizedItem(new_node.frequency, new_node))
    root_node = heapq.heappop(nodes).item
    codes = get_codes(root_node)
    encoded = "".join([codes[c] for c in data])
    return encoded, root_node


if __name__ == "__main__":
    print(encode("streets are stone stars are not"))
