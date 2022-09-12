# Copyright 2021 Xin Han
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
import string
from collections import defaultdict
from queue import Queue

import numpy as np
from numba import jit
import os


@jit(nopython=True)
def _fast_dot(x, y):
    """Compute the dot product of x and y using numba.

      Args:
      x - a numpy vector (or list).
      y - a numpy vector (or list).

      Returns:
      x_T.y
      """
    return np.dot(x, y)


class INode:
    """Isolation hc node."""

    def __init__(self):
        self.id = "id" + ''.join(random.choice(
            string.ascii_uppercase + string.digits) for _ in range(15))
        self.children = []
        self.parent = None
        self.pts = []  # each pt is a tuple of (label, id).
        self.ikv = None
        self.point_counter = 0

    def __lt__(self, other):
        """An arbitrary way to determine an order when comparing 2 nodes."""
        return self.id < other.id

    def insert(self, pt, delete_node=False, L=float("Inf"), t=200):
        """Insert a new pt into the tree.

        Apply recurse masking and balance rotations where appropriate.

        Args:
        pt - a tuple of numpy array, class label, point id.
        collapsibles - (optional) heap of collapsed nodes.
        L - (optional) maximum number of leaves in the tree.
        t - parameter of isolation kernel

        Returns:
        A pointer to the root.
        """

        if delete_node and self.point_counter >= L:
            self = self.delete()
        if self.pts is not None and len(self.pts) == 0:
            self.add_pt(pt[:2])
            self.ikv = pt[2]
            return self
        else:
            curr_node = self.root()
            x_ik = pt[2].astype(float)
            while curr_node.is_internal():
                chl_ik = curr_node.children[0].ikv.astype(float)
                chr_ik = curr_node.children[1].ikv.astype(float)
                x_dot_chl = _fast_dot(x_ik, chl_ik) / (t * math.sqrt(
                    _fast_dot(x_ik, x_ik)) * (math.sqrt(_fast_dot(chl_ik, chl_ik))))
                x_dot_chr = _fast_dot(x_ik, chr_ik) / (t * math.sqrt(
                    _fast_dot(x_ik, x_ik)) * (math.sqrt(_fast_dot(chr_ik, chr_ik))))
                if x_dot_chl >= x_dot_chr:
                    curr_node = curr_node.children[0]
                else:
                    curr_node = curr_node.children[1]
            new_leaf = curr_node._split_down(pt)
            ancs = new_leaf._ancestors()
            for a in ancs:
                a.add_pt(pt[:2])
            _ = new_leaf._update_ik_value_recursively()
            return new_leaf.root()

    def delete(self):
        curr_node = self.root()
        p_id = curr_node.pts[0]
        while curr_node.is_internal():
            curr_node.pts.remove(p_id)
            # assert (p_id in curr_node.children[0].pts[0]) != (p_id in curr_node.children[1].pts[0]), "Except: Exsiting only in  one subtree, \
            #                                                      Get: %s %s" % (p_id in curr_node.children[0].pts[0],
            #                                                                     p_id in curr_node.children[1].pts[0])
            if p_id in curr_node.children[0].pts:
                curr_node = curr_node.children[0]
            elif p_id in curr_node.children[1].pts:
                curr_node = curr_node.children[1]
        sibling = curr_node.siblings()[0]
        ancs = curr_node._ancestors()
        for a in ancs:
            a.ikv = a.ikv - curr_node.ikv
        parent_node = curr_node.parent
        if parent_node.parent:
            parent_node.parent.children.remove(parent_node)
            parent_node.parent.add_child(sibling)
        else:
            return sibling
        return self

    def _update_ik_value(self):
        """
        updata ik values

        Args:
        None.

        Returns:
        A tuple of this node and a bool that is true if the parent may need an
        update.
        """
        if self.children:
            if len(self.children) == 1:
                self.ikv = self.children[0].ikv
            else:
                self.ikv = self.children[0].ikv + self.children[1].ikv
            return self

    def _update_ik_value_recursively(self):
        """Update a node's parameters recursively.

        Args:
        None - start computation from a node and propagate upwards.

        Returns:
        A pointer to the root.
        """
        curr_node = self
        while curr_node.parent:
            _ = curr_node.parent._update_ik_value()
            curr_node = curr_node.parent
        return curr_node

    def add_child(self, new_child):
        """Add a INode as a child of this node (i.e., self).

        Args:
        new_child - a INode.

        Returns:
        A pointer to self with modifications to self and new_child.
        """
        new_child.parent = self
        self.children.append(new_child)
        return self

    def add_pt(self, pt):
        """Add a data point to this node.

        Increment the point counter. If the number of points at self is less
        than or equal to the exact distance threshold, add pt to self.pts.
        Otherwise, set self.pts to be None.

        Args:
        pt - the data point we are adding.

        Returns:
        A point to this node (i.e., self). Self now "contains" pt.
        """
        self.point_counter += 1
        if self.pts is not None:
            self.pts.append(pt)
        return self

    def _split_down(self, pt):
        """
        Create a new node for pt and a new parent with self and pt as children.

        Create a new node housing pt. Then create a new internal node. Add the
        node housing pt as a child of the new internal node. Then, disconnect
        self from its parent and make it a child of the new internal node.
        Finally, make the new internal node a child of self's old parent. Note:
        while this modifies the tree, nodes are NOT UPDATED in this procedure.

        Args:
        pt - the pt to be added.

        Returns:
        A pointer to the new node containing pt.
        """
        new_internal = INode()
        if self.pts is not None:
            new_internal.pts = self.pts[:]  # Copy points to the new node.
        else:
            new_internal.pts = None
        new_internal.point_counter = self.point_counter

        if self.parent:
            self.parent.add_child(new_internal)
            self.parent.children.remove(self)
            new_internal.add_child(self)
        else:
            new_internal.add_child(self)

        new_leaf = INode()
        new_leaf.ikv = pt[2]
        new_leaf.add_pt(pt[:2])  # This updates the points counter.
        new_internal.add_child(new_leaf)
        return new_leaf

    def purity(self, cluster=None):
        """Compute the purity of this node.

        To compute purity, count the number of points in this node of each
        cluster label. Find the label with the most number of points and divide
        bythe total number of points under this node.

        Args:
        cluster - (optional) str, compute purity with respect to this cluster.

        Returns:
        A float representing the purity of this node.
        """
        if cluster:
            pts = [p for l in self.leaves() for p in l.pts]
            return float(len([pt for pt in pts
                              if pt[0] == cluster])) / len(pts)
        else:
            label_to_count = self.class_counts()
        return max(label_to_count.values()) / sum(label_to_count.values())

    def clusters(self):
        return self.true_leaves()

    def class_counts(self):
        """Produce a map from label to the # of descendant points with label."""
        label_to_count = defaultdict(float)
        pts = [p for l in self.leaves() for p in l.pts]
        for x in pts:
            l, id = x
            label_to_count[l] += 1.0
        return label_to_count

    def pure_class(self):
        """If this node has purity 1.0, return its label; else return None."""
        cc = self.class_counts()
        if len(cc) == 1:
            return list(cc.keys())[0]
        else:
            return None

    def siblings(self):
        """Return a list of my siblings."""
        if self.parent:
            return [child for child in self.parent.children if child != self]
        else:
            return []

    def aunts(self):
        """Return a list of all of my aunts."""
        if self.parent and self.parent.parent:
            return [child for child in self.parent.parent.children
                    if child != self.parent]
        else:
            return []

    def _ancestors(self):
        """Return all of this nodes ancestors in order to the root."""
        anc = []
        curr = self
        while curr.parent:
            anc.append(curr.parent)
            curr = curr.parent
        return anc

    def depth(self):
        """Return the number of ancestors on the root to leaf path."""
        return len(self._ancestors())

    def height(self):
        """Return the height of this node."""
        return max([l.depth() for l in self.leaves()])

    def descendants(self):
        """Return all descendants of the current node."""
        d = []
        queue = Queue()
        queue.put(self)
        while not queue.empty():
            n = queue.get()
            d.append(n)
            if n.children:
                for c in n.children:
                    queue.put(c)
        return d

    def leaves(self):
        """Return the list of leaves under this node."""
        lvs = []
        queue = Queue()
        queue.put(self)
        while not queue.empty():
            n = queue.get()
            if n.children:
                for c in n.children:
                    queue.put(c)
            else:
                lvs.append(n)
        return lvs

    def true_leaves(self):
        """
        Returns all of the nodes which have no children
        (e.g. data points and collapsed nodes)
        """
        lvs = []
        queue = Queue()
        queue.put(self)
        while not queue.empty():
            n = queue.get()
            if n.children:
                for c in n.children:
                    queue.put(c)
            else:
                lvs.append(n)
        return lvs

    def lca(self, other):
        """Compute the lowest common ancestor between this node and other.

        The lowest common ancestor between two nodes is the lowest node
        (furthest distances from the root) that is an ancestor of both nodes.

        Args:
        other - a node in the tree.

        Returns:
        A node in the tree that is the lowest common ancestor between self and
        other.
        """

        ancestors = self._ancestors()
        curr_node = other
        while curr_node not in set(ancestors):
            curr_node = curr_node.parent
        return curr_node

    def root(self):
        """Return the root of the tree."""
        curr_node = self
        while curr_node.parent:
            curr_node = curr_node.parent
        return curr_node

    def is_leaf(self):
        """Returns true if self is a leaf, else false."""
        return len(self.children) == 0

    def is_internal(self):
        """Returns false if self is a leaf, else true."""
        return not self.is_leaf()
