import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation
import copy
from queue import PriorityQueue


class Graph:
    def __init__(self, vertex_num):
        self.vertex_num = vertex_num
        # 距离表
        self.edges = [[-1 for i in range(vertex_num)] for j in range(vertex_num)]

    def add_edge(self, edge):
        # 记录u，v两节点之间的距离
        # 要注意的是如果是有向图只需定义单向的权重
        # 如果是无向图则需定义双向权重
        u = edge[0]
        v = edge[1]
        weight = edge[2]
        self.edges[u][v] = weight

    def get_min_dist_vertex(self, distances, visited):
        # 遍历节点，寻找未访问的节点中距离最小者
        min_dist = float('inf')
        min_vertex = -1
        for u in range(self.vertex_num):
            if visited[u] == False and distances[u] < min_dist:
                min_dist = distances[u]
                min_vertex = u
        return min_vertex

    def dijkstra(self, start):
        # 记录每次的状态
        visited_list = []
        distances_list = []
        previous_vertex_list = []
        # 记录被访问过的节点
        visited = [False for n in range(self.vertex_num)]
        # 开始时定义源节点到其他所有节点的距离为无穷大
        # distances = {v: float('inf') for v in range(self.vertex_num)}
        distances = [float('inf') for n in range(self.vertex_num)]
        # 源节点到自己的距离为0
        distances[start] = 0
        # # 记录被访问过的节点
        # visited = []
        # 优先队列
        pq = PriorityQueue()
        pq.put((0, start))
        # 记录每个节点的前节点，便于回溯
        # previous_vertex = {}
        previous_vertex = [-1 for i in range(self.vertex_num)]

        visited_list.append(copy.copy(visited))
        distances_list.append(copy.copy(distances))
        previous_vertex_list.append(copy.copy(previous_vertex))

        while not pq.empty():
            # 得到优先级最高的节点，也就是前节点到其他节点距离最短的节点作为当前出发节点
            (dist, current_vertex) = pq.get()
            # 标记已访问过的节点(最有路径集合)
            # visited.append(current_vertex)
            visited[current_vertex] = True
            for neighbor in range(self.vertex_num):
                # 邻居节点之间距离不能为-1
                if self.edges[current_vertex][neighbor] != -1:
                    distance = self.edges[current_vertex][neighbor]
                    # 已经访问过的节点不能再次被访问
                    # if neighbor not in visited:
                    if not visited[neighbor]:
                        # 更新源节点到其他节点的最短路径
                        old_cost = distances[neighbor]
                        new_cost = distances[current_vertex] + distance
                        if new_cost < old_cost:
                            # 加入优先队列
                            pq.put((new_cost, neighbor))
                            distances[neighbor] = new_cost
                            previous_vertex[neighbor] = current_vertex
            visited_list.append(copy.copy(visited))
            distances_list.append(copy.copy(distances))
            previous_vertex_list.append(copy.copy(previous_vertex))

        return distances, previous_vertex, visited_list, distances_list, previous_vertex_list

    def shortest(self, start, target, distances, previous_vertex):
        path = []
        shortest_path = []
        key = target
        # 回溯，得到源节点到目标节点的最佳路径
        while True:
            if key == start:
                path.append(start)
                break
            else:
                path.append(key)
                key = previous_vertex[key]
        # 节点名字由数字转成字符
        for vertex in path[:: -1]:
            shortest_path.append(vertex)
        print(f"节点 {start} 到节点 {target} 的最短距离为 {distances[target]},"
              f"最短路径为 {shortest_path}")

        return shortest_path


class Draw:
    def __init__(self, edges):
        self.G = nx.Graph()
        self.G.add_weighted_edges_from(edges)
        # self.pos = nx.shell_layout(self.G)  # 用 FR算法排列节点
        self.pos = {0: (3, 4), 1: (1, 2.5), 2: (2, 1), 3: (4, 1), 4: (5, 3)}
        nx.draw(self.G, self.pos, with_labels=True, alpha=0.5, node_size=1000, font_size=20)  # 画图
        edge_labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels, font_size=20)  # 画边的权值

    def draw_path(self, path):
        edge_list = []
        for i in range(len(path) - 1):
            edge_list.append((path[i], path[i + 1]))
        nx.draw_networkx_edges(self.G, self.pos, edgelist=edge_list, edge_color='r', width=3)  # 画路径

    def draw_animation_path(self, fig, ax, n, path):
        ani = animation.FuncAnimation(fig, self.animation_path_func, frames=len(path), fargs=(n, path, ax),
                                      interval=2000)
        ani.save('path.gif')
        return ani

    def animation_path_func(self, num, n, path, ax):
        ax.clear()
        nx.draw(self.G, self.pos, with_labels=True, alpha=0.5, node_size=1000, font_size=20)
        edge_labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels, font_size=20)
        node_list = [path[0]]
        edge_list = []
        path_length = 0
        for i in range(num):
            node_list.append(path[i + 1])
            edge_list.append((path[i], path[i + 1]))
            path_length += self.G[path[i]][path[i + 1]]['weight']
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=node_list, node_color='r', node_size=1000)
        nx.draw_networkx_edges(self.G, self.pos, edgelist=edge_list, edge_color='r', width=3, arrows=True)
        ax.set_title("最短路径：{}\n最短路径长度：{}".format(node_list, path_length), fontsize=20)

    def draw_animation_process(self, fig, ax, n, visited_list, distances_list, previous_vertex_list):
        ani = animation.FuncAnimation(fig, self.animation_process_func, frames=len(visited_list),
                                      fargs=(n, visited_list, distances_list, previous_vertex_list, ax), interval=2000)
        ani.save('process.gif')
        return ani

    def animation_process_func(self, num, n, visited_list, distances_list, previous_vertex_list, ax):
        ax.clear()
        nx.draw(self.G, self.pos, with_labels=True, alpha=0.5, node_size=1000, font_size=20)
        edge_labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels, font_size=20)
        node_list = []
        for i in range(n):
            if visited_list[num][i]:
                node_list.append(i)
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=node_list, node_color='r', node_size=1000)
        ax.set_title(
            "已确定顶点数组visited：{}\n各顶点最短路径长度数组distances：{}\n各顶点最短路径前顶点数组previous：{}".format(
                node_list, distances_list[num], previous_vertex_list[num]), fontsize=20)


def main():
    n = 5
    edges = [(0, 1, 10), (0, 3, 30), (0, 4, 100), (1, 2, 50), (2, 4, 10), (3, 2, 20), (3, 4, 60)]
    start = 0
    target = 4

    # 画图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig1, ax1 = plt.subplots()
    draw = Draw(edges)
    plt.show()

    g = Graph(n)
    for edge in edges:
        g.add_edge(edge)
    distances, previous_vertex, visited_list, distances_list, previous_vertex_list = g.dijkstra(start)
    print(visited_list)
    shortest_path = g.shortest(start, target, distances, previous_vertex)

    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ani = draw.draw_animation_path(fig2, ax2, n, shortest_path)
    plt.show()
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    ani = draw.draw_animation_process(fig3, ax3, n, visited_list, distances_list, previous_vertex_list)
    plt.show()


if __name__ == '__main__':
    main()