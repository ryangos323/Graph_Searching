from math import *
from collections import deque
import random
#from disjointSet import *
from pq import PQ
import timeit

# Programming Assignment 3
# (5) After doing steps 1 through 4 below (look for relevant comments), return up here.
#     Given the output of step 4, how do the 2 versions of Dijkstra's algorithm compare?
#     How does graph density affect performance?  Does size of the graph otherwise affect performance?
#     Any other observations?
#
#
#Version 1 is always running faster than version 2.  The more dense the graph is (meaning the more edges there are),
#the farther apart the two times will be.  If the graph has the same number of vertices but less edges,
#than Version 2 will run at a time closer to version 1
#Alos, the more dense the graph is, the longer the run time will be for a particular version.  For example, version 1
#will run much quicker on a graph that has less edges, as appose to a graph with a high amount of edges
#this is consistant with the test cases provided.  In conclusion, version 1 always runs faster than version 2,
#but when the graph is dense, version 2 will be farther apart from version 1's time

# Programming Assignment 3
# (1) Implement this function, which should:
#    -- Generate a random weighted directed graph with v vertices and e different edges.
#    -- Start by generating a list of random edges (assume vertices numbered from 0 to v-1).
#       In this list, each edge is a 2-tuple, e.g., (2, 3) is an edge from vertex 2 to vertex 3.
#    -- Next, generate a list, the same length, of random weights, with minW and maxW specifying the
#       range of weights.
#    -- Then construct a Digraph object passing your lists above as parameters.  The Digraph class extends
#       Graph so actually has the same __init__ parameters as its parent class.
#    -- return that Digraph object
def generateRandomWeightedDigraph(v,e,minW,maxW) :
    edges = list()
    weights = list()
    flag = False
    for i in range(0, e):
        flag = False
        while flag == False:
            v1 = random.randint(0,v-1)
            v2 = random.randint(0,v-1)
            eTuple = (v1, v2)
            if eTuple not in edges:
                edges.append(eTuple)
                flag = True
            else:
                flag = False
    for i in range(0 , e):
        weights.append(random.randint(minW, maxW))
    WDigraph = Digraph(v, edges, weights)
    return WDigraph
    
# Programming Assignment 3
#
# (4) Make sure you find steps 2 and 3 later in this module (down in the DiGraph class) then
#     return up here to finish assignment.
#     Implement the following function to do the following:
#     -- Use your function from step (1) generate a random weighted directed graph with 16 vertices and 240 edges
#        (i.e., completely connected--all possible directed edges) and weights random in interval 1 to 10 inclusive.
#     -- Read documentation of timeit (https://docs.python.org/3/library/timeit.html)
#     -- Use timeit to time both versions of Dijkstra that you implemented in steps 2 and 3 on this graph.  The number parameter
#        to timeit controls how many times the thing you're timing is called.  To get meaningful times, you will need to experiment with this
#        a bit.  E.g., increase it if the times are too small.  Use the same value of number for timing both algorithms.
#     -- Now repeat this for a digraph with 64 vertices and 4032 edges.
#     -- Now repeat for 16 vertices and 60 edges.
#     -- Now repeat for 64 vertices and 672 edges.
#     -- Repeat this again for 16 vertices and 32 edges.
#     -- Repeat yet again with 64 vertices and 128 edges.
#    
#     -- Have this function output the timing data in a table, with columns for number of vertices, number of edges, and time.
#     -- If you want, you can include larger digraphs.  The pattern I used when indicating what size to use: Dense graphs: v, e=v*(v-1),
#        Sparse: v, e=2*v, and Something in the middle: v, e=v*(v-1)/lg V.
def timeShortestPathAlgs() :
    def alg1 ():
        times = timeit.timeit(lambda: G.DijkstrasVersion1(0), number = 1000)
        return times
    def alg2 ():
        times = timeit.timeit(lambda: G.DijkstrasVersion2(0), number = 1000)
        return times
    if __name__ == "__main__":
        L1 = []
        L2 = []
        G = generateRandomWeightedDigraph(16,240,1,10)
        d1 = alg1()
        d2 = alg2()
        L1.append(d1)
        L2.append(d2)

        G = generateRandomWeightedDigraph(64,4032,1,10)
        d1 = alg1()
        d2 = alg2()
        L1.append(d1)
        L2.append(d2)

        G = generateRandomWeightedDigraph(16,60,1,10)
        d1 = alg1()
        d2 = alg2()
        L1.append(d1)
        L2.append(d2)

        G = generateRandomWeightedDigraph(64,672,1,10)
        d1 = alg1()
        d2 = alg2()
        L1.append(d1)
        L2.append(d2)

        G = generateRandomWeightedDigraph(16,32,1,10)
        d1 = alg1()
        d2 = alg2()
        L1.append(d1)
        L2.append(d2)

        G = generateRandomWeightedDigraph(64,128,1,10)
        d1 = alg1()
        d2 = alg2()
        L1.append(d1)
        L2.append(d2)
        print("Case #\tTime 1\t\tTime 2")
        print("Case 1\t",round(L1[0],4),"\t",round(L2[0],4))
        print("Case 2\t",round(L1[1],4),"\t",round(L2[1],4))
        print("Case 3\t",round(L1[2],4),"\t",round(L2[2],4))
        print("Case 4\t",round(L1[3],4),"\t",round(L2[3],4))
        print("Case 5\t",round(L1[4],4),"\t",round(L2[4],4))
        print("Case 6\t",round(L1[5],4),"\t",round(L2[5],4))
        
# Undirected graph as adjacency lists
class Graph :
    
    # constructor
    # n is number of vertices
    def __init__(self,n=10,edges=[],weights=[]) :
        self._adj = [[] for x in range(n)]
        self._w = {}
        if len(weights) > 0 :
            for i, e in enumerate(edges) :
                self.addEdge(e[0],e[1],weights[i])
        else :
            for e in edges :
                self.addEdge(e[0],e[1])

    # adds an edge from a to b
    # For weighted graphs, w is the weight for the edge.
    # Leave the default of None for an unweighted graph.
    def addEdge(self,a,b,w=None) :
        self._adj[a].append(b)
        self._adj[b].append(a)
        if w != None :
            self._w[(a,b)] = w
            self._w[(b,a)] = w
            

    # gets number of vertices
    def numVertices(self) :
        return len(self._adj)

    # gets degree of vertex v
    def degree(self,v) :
        return len(self._adj[v])

    # BFS: s is index of starting node
    # Returns a list of VertexData objects, containing
    # distance from s (in field d) and backpointer (pred)
    def BFS(self,s) :
        class VertexData :
            pass
        vertices = [VertexData() for i in range(len(self._adj))]
        for i in range(len(vertices)) :
            vertices[i].d = inf
            vertices[i].pred = -1
        vertices[s].d = 0
        Q = deque()
        Q.append(s)
        while len(Q) > 0 :
            u = Q.popleft()
            for v in self._adj[u] :
                if vertices[v].d == inf :
                    vertices[v].d = vertices[u].d + 1
                    vertices[v].pred = u
                    Q.append(v)
        return vertices

    # DFS: Returns a list of VertexData objects containing fields for
    # discovery time (d) and finish time (f) and backpointer (pred).
    def DFS(self) :
        class VertexData :
            pass
        vertices = [VertexData() for i in range(len(self._adj))]
        for i in range(len(vertices)) :
            vertices[i].d = 0
            vertices[i].pred = -1
        time = 0
        def visit(self,u) :
            nonlocal time
            nonlocal vertices
            time = time + 1
            vertices[u].d = time
            for v in self._adj[u] :
                if vertices[v].d == 0 :
                    vertices[v].pred = u
                    visit(self,v)
            time = time + 1
            vertices[u].f = time
        
        for u in range(len(vertices)) :
            if vertices[u].d == 0 :
                visit(self,u)
        return vertices
                       
    # print graph (for testing)
    def printGraph(self) :
        print("Graph has", len(self._adj), "vertices.")
        for i, L in enumerate(self._adj) :
            print(i, "->", end="\t")
            for j in L :
                print(j, end="\t")
            print()

    def printGraphWithWeights(self) :
        print("Graph has", len(self._adj), "vertices.")
        for i, L in enumerate(self._adj) :
            print(i, "->", end="\t")
            for j in L :
                w = self._w[(i,j)]
                print(j, "(", w, ")", end="\t")
            print()

    def getEdgeList(self) :
        L = []
        for u in range(self.numVertices()) :
            for v in self._adj[u] :
                if u < v :
                    L.append((u,v))
        return L

    def MST_Kruskal(self) :
        A = set()
        DS = DisjointSets(self.numVertices())
        edges = self.getEdgeList()
        edges.sort(key=lambda e : self._w[e])
        for e in edges :
            if DS.findSet(e[0]) != DS.findSet(e[1]) :
                A.add(e)
                DS.union(e[0],e[1])
        return A

    def MST_Prim(self, r=0) :
        parent = [ None for x in range(self.numVertices())]
        Q = PQ()
        Q.add(r,0)
        for u in range(self.numVertices()) :
            if u!=r :
                Q.add(u,inf)
        while not Q.isEmpty() :
            u = Q.extractMin()
            for v in self._adj[u] :
                if Q.contains(v) and self._w[(u,v)] < Q.getPriorityValue(v) :
                    parent[v] = u
                    Q.changePriorityValue(v, self._w[(u,v)])
        A = set()
        for u, v in enumerate(parent) :
            if v!=None:
                A.add((u,v))
        return A
            
                

# Directed graph as adjacency lists
class Digraph(Graph) :            

    # adds an edge from a to be
    def addEdge(self,a,b,w=None) :
        self._adj[a].append(b)
        if w != None :
            self._w[(a,b)] = w

    def getEdgeList(self) :
        L = []
        for u in range(self.numVertices()) :
            for v in self._adj[u] :
                    L.append((u,v))
        return L

    # Programminf Assignment 3:
    # 2) Implement Dijkstra's Algorithm using a simple list as the "priority queue" as described in paragraph
    #    that starts at bottom of page 661 and continues on 662 (also described in class).
    #
    #    Have this method return a list of 3-tuples, one for each vertex, such that first position is vertex id,
    #    second is distance from source vertex (i.e., what pseudocode from textbook refers to as v.d), and third
    #    is the vertex's parent (what the textbook refers to as v.pi).  E.g., (2, 10, 5) would mean the shortest path
    #    from s to 2 has weight 10, and vertex 2's parent is vertex 5.
    #
    #    the parameter s is the source vertex.
    def DijkstrasVersion1(self,s) :
        class VertexData:
            pass
        vList = [VertexData() for i in range(len(self._adj))]
        S = list()
        Q = list()
        vList[s].dist = 0
        vList[s].prev = None
        for v in range(len(vList)):
            if(v != s):
                vList[v].dist = float('inf')
                vList[v].prev = None
            Q.append(v)
        while Q:
            u = None
            for x in Q:
                if u is None:
                    u = x
                elif vList[x].dist < vList[u].dist:
                    u = x
            if u is None:
                break
            Q.remove(u)
            S.append((u, vList[u].dist, vList[u].prev))
            for v in self._adj[u]:
                temp = vList[u].dist + self._w[(u,v)]
                if temp < vList[v].dist:
                    vList[v].dist = temp
                    vList[v].prev = u
        return S

    # Programminf Assignment 3:
    # 3) Implement Dijkstra's Algorithm using a binary heap implementation of a PQ as the PQ.
    #    Specifically, use the implementation I have posted here: https://github.com/cicirello/PythonDataStructuresLibrary
    #    Use the download link (if you simply click pq.py Github will just show you the source in a web browser with line numbers).
    #
    #    Have this method return a list of 3-tuples, one for each vertex, such that first position is vertex id,
    #    second is distance from source vertex (i.e., what pseudocode from textbook refers to as v.d), and third
    #    is the vertex's parent (what the textbook refers to as v.pi).  E.g., (2, 10, 5) would mean the shortest path
    #    from s to 2 has weight 10, and vertex 2's parent is vertex 5.
    #
    #    the parameter s is the source vertex.
    def DijkstrasVersion2(self,s) :
        class VertexData:
            pass
        vList = [VertexData() for i in range(len(self._adj))]
        S = list()
        Q = PQ()
        vList[s].dist = 0
        vList[s].prev = None
        for v in range(len(vList)):
            if(v != s):
                vList[v].dist = float('inf')
                vList[v].prev = None
            Q.add(v, vList[v].dist)
        while not Q.is_empty():
            u = Q.extract_min()
            S.append((u, vList[u].dist, vList[u].prev))
            for v in self._adj[u]:
                temp = vList[u].dist + self._w[(u,v)]
                if temp < vList[v].dist:
                    vList[v].dist = temp
                    vList[v].prev = u
                    Q.change_priority(v, temp)
        return S
    
    # Topological Sort of the directed graph (Section 22.4 from textbook).
    # Returns the topological sort as a list of vertex indices.
    #
    #       Homework Hints/Suggestions/Etc:
    #           1) Textbook indicates to use a Linked List.  Python doesn't have
    #               one in the standard library.  Instead, use a deque (don't simply use
    #               a python list since adding at the front is O(N) for a python list,
    #               while it is O(1) for a deque).
    #           2) From the pseudocode, you will be tempted to (a) call DFS, and then (b) sort
    #               vertices by the finishing times.  However, don't do that since the sort will
    #               cost O(V lg V) unnecessarily.
    #           3) So, how do you do it without sorting?
    #               A) Option A: Start by copying and pasting DFS code to start of topologicalSort, and
    #                   where finishing time is set, add the vertex index to front of list.
    #               B) Option B: Add an optional parameter to DFS method that is a
    #                   function that is called upon finishing a vertex.
    #                   Give it a default that does nothing (i.e., just a pass). Your topologicalSort would then
    #                   call DFS passing a function that adds the vertex index to the front of a list.
    #               C) Option C: any other way you can come up with that doesn't change what DFS currently does logically 
    def topologicalSort(self) :
        L = deque()
        class VertexData :
            pass
        vertices = [VertexData() for i in range(len(self._adj))]
        for i in range(len(vertices)) :
            vertices[i].d = 0
            vertices[i].pred = -1
        time = 0
        def visit(self,u) :
            nonlocal time
            nonlocal vertices
            time = time + 1
            vertices[u].d = time
            for v in self._adj[u] :
                if vertices[v].d == 0 :
                    vertices[v].pred = u
                    visit(self,v)
            time = time + 1
            vertices[u].f = time
            L.appendleft(u)
            print(L)
        
        for u in range(len(vertices)) :
            if vertices[u].d == 0 :
                visit(self,u)
        return L

    # Computes the transpose of a directed graph. (See textbook page 616 for description of transpose).
    # Does not alter the self object.  Returns a new Digraph that is the transpose of self.
    def transpose(self) :
        tranL = []
        for i in range(len(self._adj)) :
            L = self._adj[i]
            for j in range(len(L)):
                tranL.append((L[j], i))
        dGraph = Digraph(len(self._adj), tranL)
        dGraph.printGraph()
        return dGraph

    # Computes the strongly connected components of a digraph.
    # Returns a list of lists, containing one list for each strongly connected component, which is simply
    # a list of the vertices in that component.
    #
    #       Homework Hints/Suggestions/Etc: See algorithm on page 617.
    #           1) Take a look at steps 1 and 2 before you do anything.  Notice that Step 1 computes finishing times with DFS,
    #               and step 3 uses vertices in order of decreasing finishing times.  As in the topological sort, don't actually sort
    #               by finishing time (to avoid O(V lg v) step).  However, this is easier than in the topological sort as you already
    #               have a method that will get you what you need.  For step 1 of algorithm you can simply call your topological sort.
    #               That will give you the vertices in decreasing order by finishing time, which is really the intention of line 1.
    #           2) Line 2 is just the transpose and you implemented a method to compute this above.
    #           3) The DFS in line 3 can be done in a couple ways.  As above, if you change DFS, make sure it will still function in the basic
    #               version.  The simplest way to do that would be to leave it alone, and just start by copying and pasting the code.
    #               You'll need to then alter it to have the outer loop use the vertex ordering obtained from algorithm line 1 (to implement line 3).
    #               And to do line 4, you'll need to further alter it to generate the list of lists for the return value.
    def stronglyConnectedComponents(self) :
        tranSelf = self.transpose()
        tempL = []
        L = []
        d = self.topologicalSort()
        class VertexData :
            pass
        vertices = [VertexData() for i in range(len(tranSelf._adj))]
        for i in range(len(vertices)) :
            vertices[i].d = 0
            vertices[i].pred = -1
        time = 0
        def visit(tranSelf,u) :
            nonlocal time
            nonlocal vertices
            time = time + 1
            vertices[u].d = time
            tempL.append(u)
            for v in tranSelf._adj[u] :
                if vertices[v].d == 0 :
                    vertices[v].pred = u
                    visit(tranSelf,v)
            time = time + 1
            vertices[u].f = time
        
        for u in d :
            if vertices[u].d == 0 :
                visit(tranSelf,u)
                L.append(tempL)
                tempL = []
        print("\nWorking....\nWorking....\n")
        return L

# Implement any code necessary to test your topological sort, transpose, and strongly connected components methods.
#  E.g., construct graphs for the tests, figure out what the results should be, use your methods, write any code necessary to
#  output results, and check if your algorithms worked correctly.

G = Graph(7, [(0,5),(0,1),(1,3),(5,2),(5,3),(5,4),(3,6)])
H = Digraph(7, [(5,0),(0,1),(1,3),(2,5),(3,5),(5,4),(3,6)])
G2 = Graph(10, [ (x,y) for x in range(10) for y in range(x) if x % (y+1) == 0 ])

w = [           1,      2,    6,    5,    10,   4,    3,    1,    4,   3,    3,    2,    8,    4,     9,    3,    6 ]
G3 = Graph(10, [(0,1),(0,2),(1,3),(1,4),(1,7),(2,5),(2,6),(2,8),(3,7),(4,7),(5,8),(6,8),(7,9),(8,9),(0,9),(0,3),(0,6)], w)
G4 = Digraph(10, [(0,1),(0,2),(1,3),(1,4),(1,7),(2,5),(2,6),(2,8),(3,7),(4,7),(5,8),(6,8),(7,9),(8,9),(0,9),(0,3),(0,6)], w)
TEST = generateRandomWeightedDigraph(16,20,0,10)
TEST.printGraphWithWeights()
T2 = G4.DijkstrasVersion1(0)
T1 = G4.DijkstrasVersion2(0)
timeShortestPathAlgs()
print(T1)
print(T2)
