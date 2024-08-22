"""
Author : Eunice Lee Wen Jing
Date: 25/5/2024

References:
Question 2
-- Edmonds Karp Algorithm : https://www.w3schools.com/dsa/dsa_algo_graphs_edmondskarp.php#:~:text=The%20Edmonds%2DKarp%20algorithm%20works,as%20possible%20through%20that%20path.
-- BFS : https://www.w3schools.com/dsa/dsa_algo_graphs_edmondskarp.php#:~:text=The%20Edmonds%2DKarp%20algorithm%20works,as%20possible%20through%20that%20path.
-- Lecture 08 Flow Network Slide 120
-- Tutorial Studio 10 Q8 Flow Network Design

"""

# ______________________________________________________________________________________________________________________
# Question 2 : Securing the Companies

# Refer to edmonds karp algorithm online resources
class build_graph:
    """
    This class implements the Ford Fulkerson algorithm.

    We have 1 constructor and 3 methods.
    - bfs(): Performs Breadth-First Search to find the shortest path from source to sink.
    - augment_flow(): Updates the path flow found by BFS then modify the residual capacities.
    - ford_fulkerson(): Ford-Fulkerson algorithm to find the maximum flow from source to sink.
    """

    def __init__(self, graph, timetable, n_officers, m_companies):
        """
        This is a constructor.
        Initializes the build_graph attributes.

        Attributes:
        - graph: adjacency matrix for the flow network.
        - timetable: 4D list representing the security officer'source shift timetable for each company.
        allocation[i][j][d][k]
        - size: number of nodes in the graph.
        - n_officers: Number of security officers.
        - m_companies: Number of companies.

        :param graph: 2D list (adjacency matrix) for the flow network.
        :param timetable: A 4D list to store the security officer'source shift timetable for each company. [i][j][d][k]
        :param n_officers: Number of security officers.
        :param m_companies: Number of companies.

        :time complexity : O(1)
        :space complexity : O(N+M) ,
        where N represents number of officer , M represents number of companies
        """
        self.timetable = timetable
        self.graph = graph
        self.size = len(graph)
        self.n_officers = n_officers
        self.m_companies = m_companies
    # Refer to online reference
    def bfs(self, source, sink, parent):
        """
        This method is to performs BFS traversal from source to sink to find the shortest path.
        This is also an augmenting path for ford-fulkerson method.

        :param source: Source node.
        :param sink: Sink node.
        :param parent: lst, A list to store the path from source to sink.

        :return: Boolean,A boolean value showing that whether a path exists or not

        :time complexity : O(V+E)
        where V represents number of vertices (nodes in this case),
        E represents number of edges

        :space complexity : O(V)
        V number of vertices, to store visited & parent list
        """

        visited = [False] * self.size
        # Initialize BFS queue with source node
        queue = [source]
        # Source is visited
        visited[source] = True

        while queue:
            z = queue.pop(0)

            for i in range(self.size):
                # if node not visited
                if not visited[i] and self.graph[z][i] > 0:
                    queue.append(i)
                    visited[i] = True
                    parent[i] = z
                    # reach sink node
                    if i == sink:
                        return True

        return False
    # Refer to online reference
    def augment_flow(self, source, sink, parent):
        """
        This method is used to updates the path flow found by BFS then modify the residual capacities.

        Firstly, I find the minimum capacity in the path which is also the bottleneck capacity.
        Then, I reduce the capacity of forward edges and also increase the capacity of backward edges in order
        to update the residual capacities. Lastly i update the timetable [i][d][j][k] if path == 1.

        :param source: Source node.
        :param sink: Sink node.
        :param parent: lst, A list to store the path from source to sink.

        :return: int, The flow value of the path.

        :time complexity : O(V)
        where v is the max number of vertices path from souce to sink

        :space complexity: O(V)
        - This is because storage of path is proportional to V.
        """
        path_flow = float("Inf")

        s = sink
        while s != source:
            # min capacity
            path_flow = min(path_flow, self.graph[parent[s]][s])
            s = parent[s]

        v = sink
        while v != source:
            u = parent[v]
            # Update residual cap for forward edge
            self.graph[u][v] -= path_flow
            # Update residual cap for backward edge
            self.graph[v][u] += path_flow
            v = parent[v]

        v = sink
        path = []
        while v != source:
            path.append(v)
            v = parent[v]
        path.append(source)
        # Back track
        path.reverse()
        path_names = [str(node) for node in path] # 6 nodes

        if path_flow == 1:
            # Update timetable
            # 1 == Officer, 4 == Companies
            self.timetable[path[1] - 2][path[4] - 95 - self.n_officers][(path[2] - self.n_officers) // 3 - 1][(path[2] - self.n_officers) % 3] = 1
        #print(" |Path:", " -> ".join(path_names), ", Flow:", path_flow)
        return path_flow

    # Refer to lecture slide 08 Flow Network Slide 120
    def ford_fulkerson(self, source, sink):
        """
        This method is to find the max flow from source to sink using Ford-Fulkerson algorithm.

        :param source: Source node.
        :param sink: Sink node.

        :return: Maximum flow from source to sink.

        :time complexity : O(V E^2) ,
        where V represents Vertives
        E represents edges

        :space complexity : O(V), To store parent list
        """
        # Initialize parent list
        parent = [-1] * self.size
        # store max flow source to sink
        max_flow = 0

        # as long as there is augmenting path
        while self.bfs(source, sink, parent):
            # take the path
            path_flow = self.augment_flow(source, sink, parent)
            # augment flow = residual capacity
            max_flow += path_flow

        return max_flow


def allocate(preferences, officers_per_org, min_shifts, max_shifts):
    """
    This method allocates security officers to companies based on their shift preferences and the shift requirements.

    Firstly, we initialize the nodes and network. The total nodes in the flow network are:
    - 2 start nodes (real-start and fake-start)
    - n nodes for security officers
    - 90 nodes for shift-day combinations (30 days with 3 shifts every day)
    - 3 nodes for shift types (morning, noon, night)
    - m nodes for companies
    - 1 sink node

    An adjacency matrix is created for the flow network with dimensions [total_nodes][total_nodes].

    Then, we process the nodes for the flow network:
    - Set the flow from fake-start (node 1) to sink using min_shifts * n.
    - For the real-start (node 0), set the flow to each officer node to min_shifts and from fake-start (node 1) to each officer node to max_shifts - min_shifts.

    After this, we connect the officer nodes to the shift-day nodes based on officers' preferences. In total, we will have 90 nodes for shift-days.
    However, to optimize the connections, we create additional nodes for shift types.
    We connect shift-day nodes to these shift type nodes with a flow of n (number of officers).
    Then, we connect the shift type nodes (morning, noon, night) to the company nodes.
    Lastly, we connect each company node to the sink with a flow based on the total required officers for that company.
    After setting up the network, we initialize an empty timetable to store the shift allocations.
    We then use the Ford-Fulkerson algorithm to find the maximum flow from source to sink.
    We check if the total flow satisfies the requirements.

    Written by Eunice Lee Wen Jing 33250979

    :param preferences: List[List[int]], 2D list representing each officer's shift preferences.
    :param officers_per_org: List[List[int]], 2D list where each row represents a company's requirement for morning, noon, and night shifts.
    :param min_shifts: int, Minimum shifts an officer must work per month.
    :param max_shifts: int, Maximum shifts an officer can work per month.

    :return:
    - List[List[List[List[int]]]], a 4D list representing the timetable allocation[i][j][d][k], where
        i is the officer,
        j is the company,
        d is the day,
        k is the shift (0 for morning, 1 for noon, 2 for night).
    - None, if the allocation is impossible.

    :time complexity: O(MN^2)
    where n represents the number of officers, m represents the number of companies
        - Ford-Fulkerson : O(V*E^2), V: vertices/nodes, E:edges
        - In this case, V represents by the number of Officers(n), E represents by number of Companies(m)
        - Therefore, the overall time complexity : O(NM^2)

    :space complexity: O((N+M)^2)
    where n represents number of officers , m represents number of companies
     - Adjacency graph : O((N+M+96)^2)

    """
    n = len(preferences)  # Number of security officers
    m = len(officers_per_org)  # Number of companies
    shifts = len(officers_per_org[0])  # Number of shifts per company

    # Total nodes for flow network
    # 2 start nodes
    # n - security officers
    # 90 shift-day combination nodes (30 days * 3 shifts)
    # 3 shift type nodes (morning, noon, night)
    # m - companies
    # 1 sink node
    total_nodes = (1 + 1) + n + (30 * shifts) + 3 + m + (1)
    # Node indexing:
    # 0 - real-start
    # 1 - fake-start
    # 2 to n+1 - officers (total n)
    # n+2 to n+91 - shift-day combinations (total 90)
    # n+92 to n+94 - shift types (3 types)
    # n+95 to n+94+m - companies (total m)
    # n+95+m - sink

    # Initialize network graph
    network = [[0] * total_nodes for _ in range(total_nodes)]

    # Set the flow from fake-start to sink
    network[1][n + 95 + m] = min_shifts * n

    # Set the flow from real-start to each officer and from fake-start to each officer
    for i in range(n):
        network[0][2 + i] = min_shifts
        network[1][2 + i] = max_shifts - min_shifts

    # Connect officers to shift-day nodes based on preferences
    for i in range(n):
        shift1, shift2, shift3 = preferences[i]
        for j in range(30):
            if shift1:
                network[2 + i][n + 2 + j * 3 + 0] = 1
            if shift2:
                network[2 + i][n + 2 + j * 3 + 1] = 1
            if shift3:
                network[2 + i][n + 2 + j * 3 + 2] = 1

    # Connect shift-day nodes to shift type nodes
    for i in range(90):
        for j in range(3):
            network[n + 2 + i][n + 92 + j] = n

    # Connect shift type nodes to companies
    for i in range(m):
        network[n + 92][n + 95 + i] = officers_per_org[i][0] * 30
        network[n + 93][n + 95 + i] = officers_per_org[i][1] * 30
        network[n + 94][n + 95 + i] = officers_per_org[i][2] * 30

    # Connect companies to sink
    for i in range(m):
        network[n + 95 + i][n + 95 + m] = sum(officers_per_org[i]) * 30

    # Initialize the timetable
    timetable = []
    for _ in range(n):
        officer_list = []
        for _ in range(m):
            company_list = []
            for _ in range(30):
                shiftlst = [0, 0, 0]
                company_list.append(shiftlst)
            officer_list.append(company_list)
        timetable.append(officer_list)

    flow_network = build_graph(network, timetable, n, m)
    source = 1  # 在mandatory 之后的部分是1开始的
    sink = total_nodes - 1
    a1, b1 = flow_network.ford_fulkerson(source, sink), min_shifts * n

    source = 0  # mandatory
    a2, b2 = flow_network.ford_fulkerson(source, sink), min_shifts * n

    max_flow = a1 + b1

    if max_flow < sum(sum(i) for i in officers_per_org) * 30:
        return None
    else:
        return flow_network.timetable

# _____________________________________________________________________________________________________________________
# Question 2
# Test case
#if __name__ == "__main__":
#cryy T_T
# a = allocate(
#     [[1, 0, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1], [0, 0, 0], [1, 1, 0], [0, 0, 0], [0, 0, 1],
#      [0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 1]], [[4, 9, 2]],
#     20, 27)
if __name__ == "__main__":

    a = allocate(
        [[1, 1, 0], [1, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1]],
        [[7, 5, 3]], 26, 30)

    print(a)
