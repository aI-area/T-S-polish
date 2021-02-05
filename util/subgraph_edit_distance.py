#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Yijia Zheng
# @email : yj.zheng@siat.ac.cn
# @Time  : 2019/9/7 23:16

import networkx as nx
from rdkit import Chem
import queue as Q


class Entry(object):
    def __init__(self, node1, node2, w):
        self.node1 = node1
        self.node2 = node2
        self.w = w

    def __lt__(self, other):
        return self.w > other.w


# If edge type is not trival(SINGLE), add a atom to represent the edge type
# and convert all the edge type to be the same
def convert_edge_labels(g):
    G = nx.Graph()

    symbols = nx.get_node_attributes(g, 'symbol')
    bond_types = nx.get_edge_attributes(g, 'bond_type')

    num = g.number_of_nodes()

    for node in g.nodes():
        G.add_node(node, symbol=symbols[node])

    for edge in g.edges():
        first, second = edge

        bond_type = str(bond_types[first, second])
        if bond_type == 'SINGLE':
            G.add_edge(first, second)
        else:
            # if the bond is not 'SINGLE', add a node to represent the bond
            G.add_node(num, symbol=bond_type)
            G.add_edge(first, num)
            G.add_edge(num, second)
            num += 1

    return G


# TODO: Bond Stereo
def convert_edge_stereos(g):
    pass


def get_adj_list(g: nx.Graph, is_directed=False):
    adj_list = {node: [] for node in g.nodes()}
    for edge in g.edges():
        first, second = edge
        adj_list[first].append(second)
        if not is_directed:
            adj_list[second].append(first)

    return adj_list


def get_label_list(adj_list: dict, symbols: dict, all_symbols: set):
    label_list = {node: {symbol: 0 for symbol in all_symbols} for node in adj_list.keys()}
    for key, values in adj_list.items():
        for value in values:
            label_list[key][symbols[value]] += 1

    return label_list


def count_intersection(label_dict1: dict, label_dict2: dict, all_symbols: set):
    v = 0
    for symbol in all_symbols:
        v += min(label_dict1[symbol], label_dict2[symbol])

    return v


def count_union(label_dict1: dict, label_dict2: dict, all_symbols: set):
    v = 0
    for symbol in all_symbols:
        v += max(label_dict1[symbol], label_dict2[symbol])

    return v


def count_weight_matrix(g1: nx.Graph, g2: nx.Graph):
    adj_list1 = get_adj_list(g1)
    adj_list2 = get_adj_list(g2)
    symbols1 = nx.get_node_attributes(g1, 'symbol')
    symbols2 = nx.get_node_attributes(g2, 'symbol')

    all_symbols = set()
    all_symbols.update(symbols1.values())
    all_symbols.update(symbols2.values())

    label_list1 = get_label_list(adj_list1, symbols1, all_symbols)
    label_list2 = get_label_list(adj_list2, symbols2, all_symbols)

    wm = {node1: {node2: 0.01 for node2 in g2.nodes()} for node1 in g1.nodes()}
    for node1 in g1.nodes():
        for node2 in g2.nodes():
            # only node with the same symbol will have higher similarity
            if symbols1[node1] == symbols2[node2]:
                inter_value = count_intersection(label_list1[node1], label_list2[node2], all_symbols)
                union_value = count_union(label_list1[node1], label_list2[node2], all_symbols)
                wm[node1][node2] = 1 if union_value == 0 else 1 + inter_value * inter_value / union_value

    return wm


# record the nodes in g2 which the label is the same as the node in g1
def get_bipartite_list(g1, g2):
    symbols1 = nx.get_node_attributes(g1, 'symbol')
    symbols2 = nx.get_node_attributes(g2, 'symbol')

    bipartite_list = {node1: [node2 for node2 in g2.nodes()
                              if symbols1[node1] == symbols2[node2]]
                      for node1 in g1.nodes()}
    return bipartite_list


def neighbor_biased_mapper(g1, g2, bonus):
    wm = count_weight_matrix(g1, g2)
    bipartite_list = get_bipartite_list(g1, g2)
    adj_list1 = get_adj_list(g1)
    adj_list2 = get_adj_list(g2)

    prior_que = Q.PriorityQueue()
    best_mate = {}  # best mate of the node
    best_weight = {}  # best weight of the node
    for node1, values in wm.items():
        node2 = max(values, key=values.get)
        w = values[node2]
        entry = Entry(node1, node2, w)
        prior_que.put(entry)

        best_mate[node1] = node2
        best_weight[node1] = w

    mapping = {node: -1 for node in g1.nodes()}
    reverse_mapping = {node: -1 for node in g2.nodes()}

    while not prior_que.empty():
        entry = prior_que.get()

        # node1 has been mapped
        if mapping[entry.node1] != -1:
            continue

        # node2 has been mapped, find another node2
        if reverse_mapping[entry.node2] != -1:
            best_weight[entry.node1] = 0
            for node2 in bipartite_list[entry.node1]:
                if reverse_mapping[node2] < 0 and wm[entry.node1][node2] > best_weight[entry.node1]:
                    best_weight[entry.node1] = wm[entry.node1][node2]
                    best_mate[entry.node1] = node2
            if best_weight[entry.node1] > 0:
                prior_que.put(Entry(entry.node1, best_mate[entry.node1], best_weight[entry.node1]))
            continue

        # map node1 to node2
        mapping[entry.node1] = entry.node2
        reverse_mapping[entry.node2] = entry.node1

        # adjust wm[u',v'] where u' and v' are neighbors of u and v
        for node1 in adj_list1[entry.node1]:
            if mapping[node1] >= 0:
                continue
            changed = False
            for node2 in adj_list2[entry.node2]:
                if reverse_mapping[node2] >= 0:
                    continue
                wm[node1][node2] += bonus
                if wm[node1][node2] > best_weight[node1]:
                    best_weight[node1] = wm[node1][node2]
                    best_mate[node1] = node2
                    changed = True
            if changed:
                prior_que.put(Entry(node1, best_mate[node1], best_weight[node1]))

    return mapping, reverse_mapping


def count_graph_distance(g1: nx.Graph, g2: nx.Graph, mapping: dict, reverse_mapping: dict, sub: bool):
    d_vertex = 0
    d_edge = 0

    symbols1 = nx.get_node_attributes(g1, 'symbol')
    symbols2 = nx.get_node_attributes(g2, 'symbol')

    # count distance of vertices
    for node1 in g1.nodes():
        if mapping[node1] == -1:
            d_vertex += 1
        elif symbols1[node1] != symbols2[mapping[node1]]:
            d_vertex += 1
    if not sub:  # not sub graph distance
        for node2 in g2.nodes():
            if reverse_mapping[node2] == -1:
                d_vertex += 1

    # count distance of edges
    for first, second in g1.edges():
        if mapping[first] == -1 or mapping[second] == -1 or not g2.has_edge(mapping[first], mapping[second]):
            d_edge += 1
    if not sub:
        for first, second in g2.edges():
            if reverse_mapping[first] == -1 or reverse_mapping[second] == -1 \
                    or not g1.has_edge(reverse_mapping[first], reverse_mapping[second]):
                d_edge += 1

    return d_vertex + d_edge


def count_ged(g1, g2, sub=True):
    g1 = convert_edge_labels(g1)
    g2 = convert_edge_labels(g2)
    bonus = 10

    mapping, reverse_mapping = neighbor_biased_mapper(g1, g2, bonus=bonus)
    d1 = count_graph_distance(g1, g2, mapping, reverse_mapping, sub=sub)
    mapping, reverse_mapping = neighbor_biased_mapper(g2, g1, bonus=bonus)
    d2 = count_graph_distance(g2, g1, mapping, reverse_mapping, sub=sub)

    return d1 if d1 < d2 else d2


if __name__ == '__main__':
    def mol_to_nx(mol):
        G = nx.Graph()

        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(),
                       symbol=atom.GetSymbol())
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(),
                       bond.GetEndAtomIdx(),
                       bond_type=bond.GetBondType())
        return G

    # # C00152
    # smile1 = 'NC(=O)C[C@H](N)C(=O)O'
    # mol1 = Chem.MolFromSmiles(smile1)
    # g1 = mol_to_nx(mol1)

    # C00020
    smile1 = 'Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O'
    mol1 = Chem.MolFromSmiles(smile1)
    g1 = mol_to_nx(mol1)

    # C00049
    smile2 = 'N[C@@H](CC(=O)O)C(=O)O'
    mol2 = Chem.MolFromSmiles(smile2)
    g2 = mol_to_nx(mol2)

    d = count_ged(g1, g2, False)
    print('Compound 1: %s' % smile1)
    print('Compound 2: %s' % smile2)
    print('Distance: %d' % d)
