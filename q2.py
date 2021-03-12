
iterationNum = 0


class Node(object):
    def __init__(self, index, offset=0, subTotal=0, lazySubTotal=0):
        self.max = self.min = self.index = index
        self.offsetUpdated = self.subTotalUpdated = self.lazySubTotalUpdated = 0
        # Amount of nodes in the subtree starting from current one
        self._offset, self._subTotal, self._lazySubTotal = offset, subTotal, lazySubTotal
        self._left = self._right = self.parent = None

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, node):
        self._left = node
        if node:
            self.max = max(self.max, node.max)
            self.min = min(self.min, node.min)

    @property
    def right(self):
        return self._right

    @property
    def subTreeSize(self):
        return self.max - self.min + 1

    @right.setter
    def right(self, node):
        self._right = node
        if node:
            self.max = max(self.max, node.max)
            self.min = min(self.min, node.min)

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, i: int):
        if self.offsetUpdated == iterationNum:
            return

        self._offset = i
        self.offsetUpdated = iterationNum

    @property
    def subTotal(self):
        return self._subTotal

    @subTotal.setter
    def subTotal(self, i: int):
        if self.subTotalUpdated == iterationNum:
            return

        self._subTotal = int(i)
        self.subTotalUpdated = iterationNum

    @property
    def lazySubTotal(self):
        return self._lazySubTotal

    @lazySubTotal.setter
    def lazySubTotal(self, i: int):
        if self.lazySubTotalUpdated == iterationNum:
            return

        self._lazySubTotal = int(i)
        self.lazySubTotalUpdated = iterationNum

    def lazy_add(self, i, j, x):
        toAdd = (min(self.max, j) - max(self.min, i) + 1) * x
        if self.subTotalUpdated != iterationNum:
            self.subTotal += toAdd
            self.lazySubTotal += toAdd

    def display(self):
        lines, *_ = self._display_aux()
        for line in lines:
            print(line)

        print('-'*len(line))

    def _display_aux(self):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        template_print = "'%s,%s,%s,%s' % (self.index, self._offset, self._subTotal, self._lazySubTotal)"
        # No child.
        if self.right is None and self.left is None:
            line = eval(template_print)
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if self.right is None:
            lines, n, p, x = self.left._display_aux()
            s = eval(template_print)
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if self.left is None:
            lines, n, p, x = self.right._display_aux()
            s = eval(template_print)
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = self.left._display_aux()
        right, m, q, y = self.right._display_aux()
        s = eval(template_print)
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * \
            '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + \
            (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + \
            [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2


class Vaccines:
    def __init__(self, lst: list) -> None:
        size = len(lst)
        self.root = Vaccines.tree_from_sorted_list(list(range(1, size+1)))
        for _ in lst:
            pass  # TODO: Implement array of non zeroes init

    @staticmethod
    def tree_from_sorted_list(lst):
        if not lst:
            return None

        # find middle
        mid = int((len(lst)) / 2)

        # make the middle element the root
        root = Node(lst[mid])

        # left subtree of root has all
        # values <lst[mid]
        root.left = Vaccines.tree_from_sorted_list(lst[:mid])
        if root.left:
            root.left.parent = root

        root.right = Vaccines.tree_from_sorted_list(lst[mid+1:])
        if root.right:
            root.right.parent = root
        return root

    def in_order(self):
        Vaccines._in_order(self.root)
        print()

    @staticmethod
    def _in_order(root: Node) -> None:
        if root.left:
            Vaccines._in_order(root.left)
            print(root.left.index, end=" ")

        print(root.index, end=" ")

        if root.right:
            Vaccines._in_order(root.right)
            print(root.right.index, end=" ")

    def add(self, i: int, j: int, x: int) -> None:
        global iterationNum
        iterationNum += 1
        assert i <= j, f"i ({i}) must be smaller from j ({j})"

        curr: Node = self.root
        currOffset = 0
        isInRange = False

        # Left bound
        while curr:
            Vaccines.push_sum_down(curr)
            curr.subTotal += (min(curr.max, j) -
                              max(curr.min, i) + 1) * x
            # if we left/entered the range - offset handling
            if (i <= curr.index <= j) ^ isInRange:
                isInRange = not isInRange
                curr.offset += x if isInRange else -x

            if curr.index == i:
                if curr.right:
                    curr.right.subTotal += x * curr.right.subTreeSize
                    curr.right.lazySubTotal += x * curr.right.subTreeSize
                if curr.left:
                    curr.left.offset += -x
                break

            if curr.index < i:
                curr = curr.right
            else:
                right = curr.right
                if right and right.max <= j:
                    right.lazy_add(i, j, x)
                curr = curr.left

        isInRange = False
        curr: Node = self.root

        # Right bound
        while curr:
            Vaccines.push_sum_down(curr)
            curr.subTotal += (min(curr.max, j) -
                              max(curr.min, i) + 1) * x

            if (i <= curr.index <= j) ^ isInRange:
                isInRange = not isInRange
                curr.offset += x if isInRange else -x

            if curr.index == j:
                if curr.left:
                    curr.left.subTotal += x * curr.left.subTreeSize
                    curr.left.lazySubTotal += x * curr.left.subTreeSize
                if curr.right:
                    curr.right.offset += -x
                break

            # if (curr.index < j) ^ (curr.index < i):
            #     inLowerBoundSearchPath = False

            if curr.index < j:
                left = curr.left
                if left and left.min >= i:
                    left.lazy_add(i, j, x)
                curr = curr.right
            else:
                curr = curr.left

    def find(self, i: int) -> int:
        pass

    def num_of_vaccinated(self, i, j):
        global iterationNum
        iterationNum += 1
        sum = 0
        currOffset = 0

        # Find i
        curr = self.root
        while curr:
            self.push_sum_down(curr)
            currOffset += curr.offset
            # flag i's search path
            curr.lazySubTotal = curr.lazySubTotal
            # Add to sum if curr is in the range
            if i <= curr.index <= j:
                sum += currOffset

            if curr.index == i:
                break
            curr = curr.right if curr.index < i else curr.left

        iNode = curr

        # Find j
        curr = self.root
        currOffset = 0

        while curr:
            self.push_sum_down(curr)
            currOffset += curr.offset
            # flag j's search path
            curr.subTotal = curr.subTotal
            # Add to sum if curr is in the range
            if i <= curr.index <= j and curr.lazySubTotalUpdated != iterationNum:
                sum += currOffset

            if curr.index == j:
                break
            curr = curr.right if curr.index < j else curr.left

        jNode = curr

        curr = iNode
        prev = None

        # As long as we didn't find j search path
        while curr.subTotalUpdated != iterationNum:
            if curr.right and prev != curr.right:
                sum += curr.right.subTotal
            prev = curr
            curr = curr.parent

        curr = jNode
        prev = None

        while curr.lazySubTotalUpdated != iterationNum:
            if curr.left and prev != curr.left:
                sum += curr.left.subTotal

            prev = curr
            curr = curr.parent

        return sum

    @staticmethod
    def push_sum_down(curr: Node) -> None:
        if curr.lazySubTotal:
            per_node_lazy = int(curr.lazySubTotal / curr.subTreeSize)
            curr._lazySubTotal = 0

            left, right = curr.left, curr.right
            if left:
                left._lazySubTotal += per_node_lazy * left.subTreeSize
                left._subTotal += per_node_lazy * left.subTreeSize

            if right:
                right._lazySubTotal += per_node_lazy * right.subTreeSize
                right._subTotal += per_node_lazy * right.subTreeSize


if __name__ == '__main__':
    # lst = list(range(1, 16))
    # n = len(lst)
    # tree = Vaccines(lst)
    # tree.in_order()
    # tree.root.display()
    # tree.add(4, 13, 1)
    # tree.root.display()
    # print(f'Num of vaccinated: {tree.num_of_vaccinated(1,15)}')

    # tree.add(1, 6, 1)
    # tree.root.display()
    # tree.add(6, 10, 1)
    # tree.root.display()
    # tree.add(7, 12, 1)
    # tree.root.display()
    # tree.add(1, 3, 1)
    # tree.root.display()

    # print(f'Num of vaccinated: {tree.num_of_vaccinated(1,15)}')
    # print(f'Num of vaccinated: {tree.num_of_vaccinated(1,7)}')
    # print(f'Num of vaccinated: {tree.num_of_vaccinated(6,11)}')
    # tree.add(6, 11, 1)
    # tree.root.display()
    # print(f'Num of vaccinated: {tree.num_of_vaccinated(6,11)}')
    # tree.root.display()

    import random as rand

    # Tests count
    for _ in range(1):
        size = rand.randint(5, 200)
        inefficient = [0]*size
        tree = Vaccines(inefficient)
        print(inefficient)

        # Ranges count
        for _ in range(rand.randint(1, 500)):
            i = rand.randint(1, size)
            j = rand.randint(i, size)
            x = rand.randint(1, 11)
            tree.add(i, j, x)
            inefficient = (inefficient[:i-1] +
                           [t + x for t in inefficient[i-1:j]]
                           + inefficient[j:])
            print(f'i:{i} ; j:{j} ; x:{x}')
            print(inefficient)
            trueSum = sum(inefficient[i-1:j])
            mySum = tree.num_of_vaccinated(i,j)
            assert mySum == trueSum, f'My sum is: {mySum} should be {trueSum}'
