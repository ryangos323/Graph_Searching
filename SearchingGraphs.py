from math import *
from collections import deque
import random
#from disjointSet import *
from pq import PQ
import timeit

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

    # Computes the transpose of a directed graph.
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
