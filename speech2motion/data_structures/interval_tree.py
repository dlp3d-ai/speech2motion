from typing import Any, Union

from ..utils.super import Super


class IntervalUnavailableError(Exception):
    """Exception raised when interval is unavailable.

    This exception is raised when attempting to insert a node into the interval tree
    but cannot find a suitable position.
    """
    pass


class IntervalOutOfRangeError(Exception):
    """Exception raised when interval is out of range.

    This exception is raised when attempting to insert a node that exceeds
    the range of its parent node.
    """
    pass

class IntervalExtendError(Exception):
    """Exception raised when interval extension fails.

    This exception is raised when attempting to extend a node's range
    but the operation is invalid.
    """
    pass

class IntervalTreeNode(Super):
    """Tree node for interval tree.

    Used to build and manage interval tree data structure, supporting
    node insertion, querying, and range operations.
    """

    def __init__(self,
                 start: float | int,
                 end: float | int,
                 payload: Any | None = None,
                 logger_cfg: None | dict = None) -> None:
        """Initialize interval tree node.

        Args:
            start (float | int):
                Start value of the interval node.
            end (float | int):
                End value of the interval node.
            payload (Any | None, optional):
                Payload of the interval node. Defaults to None.
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed description.
                Logger name will use the class name. Defaults to None.
        """
        Super.__init__(self, logger_cfg=logger_cfg)
        self.start = start
        self.end = end
        self.payload = payload
        self.lchild = None
        self.rchild = None
        self.parent = None
        self.side = None

    def is_leaf(self) -> bool:
        """Check if node is a leaf node.

        Leaf nodes are nodes without children and with payload.

        Returns:
            bool: True if the node is a leaf node.
        """
        return self.payload is not None

    def is_empty(self) -> bool:
        """Check if node is an empty node.

        Empty nodes are nodes without children and without payload.

        Returns:
            bool: True if the node is an empty node.
        """
        return self.payload is None and self.lchild is None and self.rchild is None

    def set_child(self, child_node: 'IntervalTreeNode', side: str) -> None:
        """Set child node for current node.

        Args:
            child_node (IntervalTreeNode):
                Child node to be set.
            side (str):
                Position of the child node. 'l' for left child, 'r' for right child.

        Raises:
            ValueError:
                If side parameter is not 'l' or 'r'.
        """
        assert self.payload is None
        child_node.parent = self
        child_node.side = side
        if side == 'l':
            self.lchild = child_node
        elif side == 'r':
            self.rchild = child_node
        else:
            msg = f'Invalid side parameter: {side}.'
            self.logger.error(msg)
            raise ValueError(msg)

    def get_next_blank_recursively(
            self,
            intervals_to_skip: list[tuple[float, float]] | None = None
            ) -> Union['IntervalTreeNode', None]:
        """Get next blank node in the tree recursively.

        Args:
            intervals_to_skip (list[tuple[float, float]], optional):
                List of intervals to skip. Returned interval should not overlap
                with intervals in this list. Defaults to None, no restriction
                on return value.

        Returns:
            Union[IntervalTreeNode, None]:
                Next blank node in the tree. Returns None if no blank node
                is found.
        """
        # no blank under self
        if self.is_leaf():
            return None
        # self is blank
        elif self.lchild is None and self.rchild is None:
            # check if self is in intervals_to_skip
            if intervals_to_skip is not None:
                for interval in intervals_to_skip:
                    if self.start >= interval[0] and self.start < interval[1]:
                        return None
                    if self.end > interval[0] and self.end <= interval[1]:
                        return None
            return self
        else:
            # search left subtree first
            if self.lchild is not None:
                left_black_node = self.lchild.get_next_blank_recursively(
                    intervals_to_skip)
                if left_black_node is not None:
                    return left_black_node
            if self.rchild is not None:
                right_black_node = self.rchild.get_next_blank_recursively(
                    intervals_to_skip
                )
                if right_black_node is not None:
                    return right_black_node
            # neither left nor right has blank node
            return None

    def get_first_leaf(self) -> dict | None:
        """Get first leaf node in the tree.

        Returns:
            dict | None:
                Dictionary containing leaf node information with start, end
                and payload fields. Returns None if no leaf node is found.
        """
        if self.is_leaf():
            node_dict = dict(
                start=self.start,
                end=self.end,
                payload=self.payload)
            return node_dict
        if self.lchild is not None:
            lchild_leaf = self.lchild.get_first_leaf()
            if lchild_leaf is not None:
                return lchild_leaf
        if self.rchild is not None:
            rchild_leaf = self.rchild.get_first_leaf()
            if rchild_leaf is not None:
                return rchild_leaf
        return None

    def get_last_leaf(self) -> Union['IntervalTreeNode', None]:
        """Get last leaf node in the tree.

        Returns:
            Union[IntervalTreeNode, None]:
                Dictionary containing leaf node information with start, end
                and payload fields. Returns None if no leaf node is found.
        """
        if self.is_leaf():
            node_dict = dict(
                start=self.start,
                end=self.end,
                payload=self.payload)
            return node_dict
        if self.rchild is not None:
            rchild_leaf = self.rchild.get_last_leaf()
            if rchild_leaf is not None:
                return rchild_leaf
        if self.lchild is not None:
            lchild_leaf = self.lchild.get_last_leaf()
            if lchild_leaf is not None:
                return lchild_leaf
        return None

    def to_list(self) -> list[dict]:
        """Convert tree to ordered list.

        Returns:
            list[dict]: List of dictionaries representing the tree.
        """
        # one element list
        if self.is_leaf():
            node_dict = dict(
                start=self.start,
                end=self.end,
                payload=self.payload)
            return [
                node_dict,
            ]
        # list of multiple leaves
        else:
            # check if self is a blank
            if self.lchild is None and self.rchild is None:
                return []
            left_list = self.lchild.to_list()
            right_list = self.rchild.to_list()
        return left_list + right_list

    def to_string(self) -> str:
        """Convert tree to string format graph.

        Each node's start value, end value, and leaf node status are included.

        Returns:
            str: Multi-line string graph of the tree.
        """
        lines = []

        def _graph_traverse(lines: list[str], node: 'IntervalTreeNode',
                            prefix: str) -> None:
            if node is not None:
                lines.append(f'{prefix}Node(start={node.start}, ' +
                             f'end={node.end}, ' +
                             f'is_leaf={node.is_leaf()})')
                if node.lchild or node.rchild:
                    if node.lchild:
                        _graph_traverse(lines, node.lchild, prefix + '  L-')
                    else:
                        lines.append(f'{prefix}  L-None')
                    if node.rchild:
                        _graph_traverse(lines, node.rchild, prefix + '  R-')
                    else:
                        lines.append(f'{prefix}  R-None')

        _graph_traverse(lines, self, '')
        return '\n'.join(lines)

    def insert_node_recursively(self,
                                node_to_insert: 'IntervalTreeNode') -> None:
        """Insert node into tree recursively based on start and end values.

        Args:
            node_to_insert (IntervalTreeNode):
                Node to be inserted.

        Raises:
            IntervalUnavailableError:
                If node cannot be inserted.
            IntervalOutOfRangeError:
                If node is out of range.
        """
        if self.is_leaf():
            msg = 'Cannot insert node under leaf node.\n' +\
                f'node_to_insert: {node_to_insert.to_string()}\n' +\
                f'self: {self.to_string()}'
            self.logger.error(msg)
            raise IntervalUnavailableError(msg)
        elif node_to_insert.start < self.start or \
                node_to_insert.end > self.end:
            msg = 'Cannot insert node out of range.\n' +\
                f'node_to_insert: {node_to_insert.to_string()}\n' +\
                f'self: {self.to_string()}'
            self.logger.error(msg)
            raise IntervalOutOfRangeError(msg)
        else:
            n_none_children = 0
            if self.lchild is None:
                n_none_children += 1
            if self.rchild is None:
                n_none_children += 1
            if n_none_children == 0 or n_none_children == 1:
                split_point = self.lchild.end if self.lchild is not None \
                    else self.rchild.start
                # check if it fits left child
                if node_to_insert.end <= split_point:
                    if self.lchild is not None:
                        self.lchild.insert_node_recursively(node_to_insert)
                    else:
                        self.set_child(node_to_insert, 'l')
                # check if it fits right child
                elif node_to_insert.start >= split_point:
                    if self.rchild is not None:
                        self.rchild.insert_node_recursively(node_to_insert)
                    else:
                        self.set_child(node_to_insert, 'r')
                else:
                    msg = 'Cannot insert node, insufficient space.\n' +\
                        f'node_to_insert: {node_to_insert.to_string()}\n' +\
                        f'self: {self.to_string()}'
                    self.logger.error(msg)
                    raise IntervalUnavailableError(msg)
            # both left and right child are None, create a new
            # child node for node_to_insert
            else:
                if self.start == node_to_insert.start:
                    # self's range == node_to_insert's range
                    # replace self's attr with node_to_insert
                    if self.end == node_to_insert.end:
                        self.logger = node_to_insert.logger
                        self.start = node_to_insert.start
                        self.end = node_to_insert.end
                        self.payload = node_to_insert.payload
                        self.lchild = node_to_insert.lchild
                        self.rchild = node_to_insert.rchild
                        # self's parent and side remain unchanged
                    # insert node_to_insert as self's lchild
                    else:
                        self.set_child(node_to_insert, 'l')
                        self.set_child(
                            IntervalTreeNode(
                                start=node_to_insert.end,
                                end=self.end,
                                logger_cfg=self.logger_cfg),
                            'r')
                # insert node_to_insert as self's rchild
                elif self.end == node_to_insert.end:
                    self.set_child(
                        IntervalTreeNode(
                            start=self.start,
                            end=node_to_insert.start,
                            logger_cfg=self.logger_cfg),
                        'l')
                    self.set_child(node_to_insert, 'r')
                # split self by node_to_insert.end
                # and insert into self's lchild
                else:
                    self.set_child(
                        IntervalTreeNode(
                            start=self.start,
                            end=node_to_insert.end,
                            logger_cfg=self.logger_cfg),
                        'l')
                    self.set_child(
                        IntervalTreeNode(
                            start=node_to_insert.end,
                            end=self.end,
                            logger_cfg=self.logger_cfg),
                        'r')
                    self.lchild.insert_node_recursively(node_to_insert)

    def extend_start(self, new_start: float | int) -> None:
        """Extend node's start value to the left, preserving all leaf nodes.

        Args:
            new_start (float | int):
                New start value of the node.

        Raises:
            IntervalExtendError:
                If node cannot be extended to the left.
        """
        if new_start > self.start:
            msg = ('Cannot extend node start value to the left, '
                   'new start value is greater than current start value. '
                   f'new_start={new_start}, self.start={self.start}')
            self.logger.error(msg)
            raise IntervalExtendError(msg)
        elif new_start == self.start:
            msg = ('Cannot extend node start value to the left, '
                   'new start value equals current start value. '
                   f'new_start={new_start}, self.start={self.start}')
            self.logger.warning(msg)
            return
        old_lchild = self.lchild
        old_rchild = self.rchild
        self.start = new_start
        self.lchild = None
        self.rchild = None
        if old_lchild is not None and not old_lchild.is_empty():
            self.insert_node_recursively(old_lchild)
        if old_rchild is not None and not old_rchild.is_empty():
            self.insert_node_recursively(old_rchild)

    def extend_end(self, new_end: float | int) -> None:
        """Extend node's end value to the right, preserving all leaf nodes.

        Args:
            new_end (float | int):
                New end value of the node.

        Raises:
            IntervalExtendError:
                If node cannot be extended to the right.
        """
        if new_end < self.end:
            msg = ('Cannot extend node end value to the right, '
                   'new end value is less than current end value. '
                   f'new_end={new_end}, self.end={self.end}')
            self.logger.error(msg)
            raise IntervalExtendError(msg)
        elif new_end == self.end:
            msg = ('Cannot extend node end value to the right, '
                   'new end value equals current end value. '
                   f'new_end={new_end}, self.end={self.end}')
            self.logger.warning(msg)
            return
        old_lchild = self.lchild
        old_rchild = self.rchild
        self.end = new_end
        self.lchild = None
        self.rchild = None
        if old_lchild is not None and not old_lchild.is_empty():
            self.insert_node_recursively(old_lchild)
        if old_rchild is not None and not old_rchild.is_empty():
            self.insert_node_recursively(old_rchild)

    def shallow_copy(self) -> 'IntervalTreeNode':
        """Create shallow copy of the node.

        Returns:
            IntervalTreeNode:
                Shallow copy result of the node. Child nodes still reference
                the original node's child nodes.
        """
        ret_node = IntervalTreeNode(
            start=self.start,
            end=self.end,
            payload=self.payload,
            logger_cfg=self.logger_cfg)
        ret_node.lchild = self.lchild
        ret_node.rchild = self.rchild
        ret_node.parent = self.parent
        ret_node.side = self.side
        return ret_node
