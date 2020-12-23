from tkinter import *
import random
import queue
import copy
import time
import plotly.graph_objects as go
import timeit
import numpy as np



class Node:
    def __init__(self,value,node_list,i,j):
        self.value= value
        self.node_list=node_list
        self.shortest_path=0
        self.i=i
        self.j=j
    def __lt__(self,other):
        return True


class Directed_Graph:
    def __init__(self,build_array):
        n=len(build_array)
        self.eval=0
        self.mat_of_nodes=[]
        self.time=0
        self.iter=0
        for i in range(n):
            y=[]
            for j in range(n):
                y.append(Node(build_array[i][j],None,i,j))
            self.mat_of_nodes.append(y)
        for i in range(n):
            for j in range(n):
                adj_mat=[]
                if i+self.mat_of_nodes[i][j].value<=n-1:
                    adj_mat.append(self.mat_of_nodes[i+self.mat_of_nodes[i][j].value][j])
                if i-self.mat_of_nodes[i][j].value>=0:
                    adj_mat.append(self.mat_of_nodes[i-self.mat_of_nodes[i][j].value][j])
                if j+self.mat_of_nodes[i][j].value<=n-1:
                    adj_mat.append(self.mat_of_nodes[i][j+self.mat_of_nodes[i][j].value])
                if j-self.mat_of_nodes[i][j].value>=0:
                    adj_mat.append(self.mat_of_nodes[i][j-self.mat_of_nodes[i][j].value])
                self.mat_of_nodes[i][j].node_list=adj_mat

    def fix(self):
        n=len(self.mat_of_nodes)
        self.eval=0
        matcopy=copy.deepcopy(self)
        self.mat_of_nodes=[]
        for i in range(n):
            y=[]
            for j in range(n):
                y.append(matcopy.mat_of_nodes[i][j])
            self.mat_of_nodes.append(y)
        for i in range(n):
            for j in range(n):
                adj_mat=[]
                if i+self.mat_of_nodes[i][j].value<=n-1:
                    adj_mat.append(self.mat_of_nodes[i+self.mat_of_nodes[i][j].value][j])
                if i-self.mat_of_nodes[i][j].value>=0:
                    adj_mat.append(self.mat_of_nodes[i-self.mat_of_nodes[i][j].value][j])
                if j+self.mat_of_nodes[i][j].value<=n-1:
                    adj_mat.append(self.mat_of_nodes[i][j+self.mat_of_nodes[i][j].value])
                if j-self.mat_of_nodes[i][j].value>=0:
                    adj_mat.append(self.mat_of_nodes[i][j-self.mat_of_nodes[i][j].value])
                self.mat_of_nodes[i][j].node_list=adj_mat


def breadth_first(nmat):
    starttime = timeit.default_timer()


    mat=nmat.mat_of_nodes
    mat_start=mat[0][0]
    n=len(mat)
    q = queue.Queue()
    visited=[]
    start=1
    q.put(mat_start)
    path_count=1
    while q.empty()!=True:
        if(start==1):
            node=q.get(True,None)
            visited.append(node)
            for i in range(len(node.node_list)):
                q.put(node.node_list[i])
                visited.append(node.node_list[i])
                node.node_list[i].shortest_path=node.shortest_path+1
            start=0
        else:
            node=q.get(True,None)
            for i in range(len(node.node_list)):
                if node.node_list[i] not in visited:
                    q.put(node.node_list[i])
                    visited.append(node.node_list[i])
                    node.node_list[i].shortest_path=node.shortest_path+1
    if mat[n-1][n-1].shortest_path==0:
        nmat.eval=(-1)*(n*n-len(visited))
    else:
        nmat.eval=mat[n-1][n-1].shortest_path

    thetime=timeit.default_timer()-starttime
    return thetime
            

def hill_climbing(iter,n):
    thetime=0  


    origmat=create_game(n)

    starttime = timeit.default_timer()

    breadth_first(origmat)
    funval=copy.deepcopy(origmat.eval)
    i=0
    while(i<iter):
        newmat=copy.deepcopy(origmat) #can't equal origmat, needs to be a copy not same memory location
        x=random.randint(0,n-1)
        y=random.randint(0,n-1)
        while x==n-1 and y==n-1:
            x=random.randint(0,n-1)
            y=random.randint(0,n-1)
        max_list=[n-(x+1),x,n-(y+1),y]
        max_val=max(max_list)
        rand_val=random.randint(1,max_val)
        newmat.mat_of_nodes[x][y].value=rand_val
        newmat.fix()
        breadth_first(newmat)
        newfunval=copy.deepcopy(newmat.eval)

        #print(origmat.mat_of_nodes[x][y].value)
        #print(newmat.mat_of_nodes[x][y].value)
        
        if funval<=newfunval:
            origmat=newmat
        i=i+1
    thetime=timeit.default_timer()-starttime
    origmat.time=thetime
    #print(funval)
    #print(origmat.eval)
    show_game2(origmat)
    return origmat
    #show_game2(origmat)


def Astar(n): 
    nmat=create_game(n)
    thetime=0  
    #copynmat=copy.deepcopy(nmat)
    #breadth_first(copynmat)
    starttime = timeit.default_timer()

    mat=nmat.mat_of_nodes
    mat_start=[0,mat[0][0]]
    pqueue=queue.PriorityQueue()
    visited=[]
    solution=None
    pqueue.put(mat_start)

    while pqueue.empty()!=True:
        node=pqueue.get(True,None)
        if node[1].i==n-1 and node[1].j==n-1:
            solution=node
            break
        visited.append(node)
        for i in range(len(node[1].node_list)):
            jchild_node=node[1].node_list[i]
            g=node[1].shortest_path+1
            fval=g+heuristic(jchild_node,mat)

            if(jchild_node.shortest_path==0):
                jchild_node.shortest_path=g
            if jchild_node.shortest_path>g:
                jchild_node.shortest_path=g
            child_node=[fval,jchild_node]
            c=0
            z=0
            foundnode=[0,Node(0,None,0,0)]
            foundnode2=[0,Node(0,None,0,0)]

            for x in visited:
                if jchild_node==x[1]:
                    c=1
                    foundnode=x
            for y in pqueue.queue:
                if jchild_node==y[1]:
                    z=1
                    foundnode2=y
            if c==0 and z==0:
                pqueue.put(child_node)
                if(fval==g):
                    continue
            elif z==1:
                if child_node[0]<foundnode2[0]:
                    foundnode2[0]=child_node[0]
                    if foundnode2[1].i==n-1 and foundnode2[1].j==n-1:
                        solution=foundnode2
    
    if solution:
        thetime=timeit.default_timer()-starttime
        nmat.eval=solution[0]
        nmat.time=thetime
        #print(nmat.eval)
        #show_game(copynmat)
        return nmat
    else:
        thetime=timeit.default_timer()-starttime
        nmat.eval=(-1)*(n*n-len(visited))
        nmat.time=thetime
        #print(nmat.eval)
        #show_game(copynmat)
        return nmat







def heuristic(child_node,mat):
    val=child_node.value
    i=child_node.i
    j=child_node.j
    #print("i="+str(i))
    #print("j="+str(j))
    n=len(mat)
    #print("n="+str(n))
    funval=0
    if i==(n-1) and j==(n-1):
        #print("were at the end")
        funval=0
    elif i==(n-1) or j==(n-1):
        #print("whats up1")
        funval=1
    elif i+val==(n-1) or j+val==(n-1):
        #print("whats up2")
        funval=2
    else:
        #print("whats up3")
        funval=3
    #print("funval="+str(funval))
    return funval


def geneticAlgo(z,n):
    gameArray = []
    valid=False
    

    counter=0
    while(counter!=z):
        p= create_game(n)
        
        gameArray.append(p)
        counter =counter + 1
    counter =0
    eval= []
    while(counter!=z):
            breadth_first(gameArray[counter])
            eval.append(gameArray[counter].eval)
            counter = counter +1
    #show_game(gameArray[1])
    counter =0
    x=0
    z=0
    start = time.time()
    selected=[]
    eval.sort(reverse=True)
    if(eval[0]>0 and eval[1]>0):
         found=False
         y=0
         while(found == False):
            if(gameArray[counter].eval==eval[0]):
                if(x<1):
                    selected.append(gameArray[counter])
                    y= y+1
                    x= x+1
                    counter= counter+1
                    if(y>1):
                        counter= counter-1
                        found=True
                    if(gameArray[counter].eval==eval[1]):
                       selected.append(gameArray[counter]) 
                       y= y+1
                    
            if(gameArray[counter].eval==eval[1]):
                if(z<1):
                    selected.append(gameArray[counter])
                    y= y+1
                    z= z+1
                    if(y>1):
                        counter=counter-1
                        found =True
                    counter= counter+1
                    if(gameArray[counter].eval==eval[0]):
                        selected.append(gameArray[counter])
                        y=y+1

                    
            if(y>1):
                found=True 
            else:
                counter= counter +1

    

    else:
        geneticAlgo(z)
   
    game_mutate=[]
    game_mutate.append(mutate_game(n,selected[0],selected[1]))
    game_mutate.append(mutate_game(n,selected[0],selected[1]))
    game_mutate.append(mutate_game(n,game_mutate[0],game_mutate[1]))
    
    
    evolve=False 
    MTC=3
    print("we here")
    while(evolve==False):
        game_mutate.append(mutate(1,n,game_mutate[MTC-1]))
        breadth_first(game_mutate[MTC])
        if(game_mutate[MTC].eval<=game_mutate[MTC-1].eval and game_mutate[MTC].eval>0):
            game_mutate.append(mutate_game(n,game_mutate[MTC],game_mutate[MTC-1]))
            MTC=MTC+1

           # print"smaller than previous kids"
        if(game_mutate[MTC].eval<=0 ):
            #print" deleted"
            del game_mutate[MTC]
            MTC= MTC-1
            game_mutate.append(mutate_game(n,game_mutate[MTC],game_mutate[MTC-1]))
            MTC= MTC +1
        if(game_mutate[MTC].eval<=selected[0].eval or game_mutate[MTC].eval<=selected[1].eval):
            if(selected[0].eval>selected[1].eval):
                game_mutate.append(mutate_game(n,game_mutate[MTC],selected[0]))
            if(selected[1].eval>selected[0].eval):
                game_mutate.append(mutate_game(n,game_mutate[MTC],selected[1]))
            if(selected[1].eval==selected[0].eval):
                game_mutate.append(mutate_game(n,game_mutate[MTC],selected[0]))

            MTC = MTC + 1
           # print "smaller than parents"

        else:
            evolve=True
    print("we done")
    end = time.time()
    game_mutate[MTC].iter=(MTC-3)
    game_mutate[MTC].time=end-start



    return game_mutate[MTC]
    #show_game(game_mutate[MTC])

def compare():
    time =False
    while(time==False):
        compare=[]
        compare.append(geneticAlgo(50))
        compare.append(hill_climbing(50, 5))
        print(compare[0].time)
        print(compare[1].time)
        if abs(compare[0].time-compare[1].time)/compare[1].time<.1:
            time= True
            print("found equal time")
            print("genetic algo eval")
            print(compare[0].eval)
            print("hill climbing eval")
            print(compare[1].eval)
            show_game3(compare[0],compare[1])



def mutate(iter,n,newGame):
    newmat=copy.deepcopy(newGame) #can't equal origmat, needs to be a copy not same memory location
    x=random.randint(0,n-1)
    y=random.randint(0,n-1)
    while x==n-1 and y==n-1:
        x=random.randint(0,n-1)
        y=random.randint(0,n-1)
    max_list=[n-(x+1),x,n-(y+1),y]
    max_val=max(max_list)
    rand_val=random.randint(1,max_val)
    newmat.mat_of_nodes[x][y].value=rand_val
    newmat.fix()
    breadth_first(newmat)    
    return newmat

def mutate_game(n,game1,game2):
    game_mutate=[]
    for i in range(n): #Rows
        x=[]
        for j in range(n): #Columns
            if i==n-1 and j==n-1:
                x.append(0)
            else:
                max_list=[n-(i+1),i,n-(j+1),j]
                max_val=max(max_list)
                rand_val=random.randint(0,1)
                if(rand_val==1):
                    x.append(game1.mat_of_nodes[i][j].value)
                if(rand_val==0):
                    x.append(game2.mat_of_nodes[i][j].value)
        game_mutate.append(x)

    offspring_Game=Directed_Graph(game_mutate)
    return offspring_Game









#
#
#

def create_game(n):
    game_mat=[]
    for i in range(n): #Rows
        x=[]
        for j in range(n): #Columns
            if i==n-1 and j==n-1:
                x.append(0)
            else:
                max_list=[n-(i+1),i,n-(j+1),j]
                max_val=max(max_list)
                rand_val=random.randint(1,max_val)
                x.append(rand_val)
        game_mat.append(x)

    game_nodes=Directed_Graph(game_mat)
    return game_nodes

def show_game(game_nodes):
    root = Tk()
    n=len(game_nodes.mat_of_nodes)
    for i in range(n): #Rows
        root.columnconfigure(i, weight=1, minsize=50)
        root.rowconfigure(i, weight=1, minsize=50)
        for j in range(n): #Columns
            if i==n-1 and j==n-1:
                b = Text(master=root)
                b.insert("1.0",0)
                sp=game_nodes.mat_of_nodes[i][j].shortest_path
                if sp==0 and (i,j)!=(0,0):
                    sp="X"
                b.insert("2.0","\nShortest Path={}\nEval Function={}".format(sp,game_nodes.eval))
                b.grid(row=i, column=j)
            else:
                val= game_nodes.mat_of_nodes[i][j].value
                b = Text(master=root)
                b.insert("1.0",val)
                sp=game_nodes.mat_of_nodes[i][j].shortest_path
                if sp==0 and (i,j)!=(0,0):
                    sp="X"
                b.insert("2.0","\nShortest Path={}\n".format(sp))
                b.grid(row=i, column=j)

    mainloop()  

def show_game2(game_nodes):
    root = Tk()
    n=len(game_nodes.mat_of_nodes)
    for i in range(n): #Rows
        root.columnconfigure(i, weight=1, minsize=50)
        root.rowconfigure(i, weight=1, minsize=50)
        for j in range(n): #Columns
            if i==n-1 and j==n-1:
                b = Text(master=root)
                b.insert("1.0",0)
                sp=game_nodes.mat_of_nodes[i][j].shortest_path
                if sp==0 and (i,j)!=(0,0):
                    sp="X"
                b.insert("2.0","\nShortest Path={}\nEval Function={}\nTime={}".format(sp,game_nodes.eval,game_nodes.time))
                b.grid(row=i, column=j)
            else:
                val= game_nodes.mat_of_nodes[i][j].value
                b = Text(master=root)
                b.insert("1.0",val)
                sp=game_nodes.mat_of_nodes[i][j].shortest_path
                if sp==0 and (i,j)!=(0,0):
                    sp="X"
                b.insert("2.0","\nShortest Path={}\n".format(sp))
                b.grid(row=i, column=j)



    mainloop() 

def hillplot(n):
    avg_array=[]
    k=50
    for j in range(k):
        avg=0
        for i in range(50):
            evalu=hill_climbing(j,n).eval
            avg=avg+evalu
        avg=avg/50
        avg_array.append(avg)

    x=np.arange(k)
    fig=go.Figure(data=go.Scatter(x=x,y=avg_array))
    fig.update_layout(
    title="Hill Climbing",
    xaxis_title="Number of Iterations",
    yaxis_title="Evaluation Function")
    fig.show()
    print(avg_array)


def spfplotsize():
    timearray=[]
    sizearray=[5,7,9,11]
    iter=50
    for k in sizearray:
        game_array=[]
        for i in range(iter):
            game=create_game(k)
            game_array.append(game)
      
        thetime=0

        for j in range(iter):  


            thetime=breadth_first(game_array[j])+thetime



        thetime=thetime/iter
        timearray.append(thetime)
    print(timearray)
    fig=go.Figure(data=go.Scatter(x=sizearray,y=timearray))
    fig.update_layout(
    title="Shortest Path First",
    xaxis_title="Size n (nxn)",
    yaxis_title="Execution Time (s)")
    fig.show()

    

def spfplotdiff():
    info={}
    sizearray=[5,7,9,11]
    for h in sizearray:
        for i in range(1000):
            game=create_game(h)

    
            thetime=breadth_first(game)

            difficulty=game.eval
            if difficulty in info:
                time=info[difficulty][0]+thetime
                number=info[difficulty][1]+1
                info[difficulty]=[time,number]
            else:
                info[difficulty]=[thetime,1]
    
    x=[]
    for j in info:
        x.append(j)
    x.sort()
    y=[]
    for k in x:
        y.append(info[k][0]/info[k][1])
    print(x)
    fig=go.Figure(data=go.Scatter(x=x,y=y))
    fig.update_layout(
    title="Shortest Path First",
    xaxis_title="Difficulty (aka eval function)",
    yaxis_title="Execution Time (s)")
    fig.show()


def hillplotsize():
    timearray=[]
    sizearray=[5,7,9,11]
    iter=50
    for k in sizearray:
        game_array=[]
        for i in range(iter):
            gamelen=k
            game_array.append(gamelen)
      
        thetime=0

        for j in range(iter):  
    

            thetime=hill_climbing(50,gamelen).time+thetime

        

        thetime=thetime/iter
        timearray.append(thetime)
    print(timearray)
    fig=go.Figure(data=go.Scatter(x=sizearray,y=timearray))
    fig.update_layout(
    title="Hill Climbing",
    xaxis_title="Size n (nxn)",
    yaxis_title="Execution Time (s)")
    fig.show()


def hillplotdiff():
    info={}
    sizearray=[5,7,9,11]
    for h in sizearray:
        for i in range(1000):

        
            difficulty=hill_climbing(10,h).eval
            thetime=hill_climbing(10,h).time
            

            if difficulty in info:
                time=info[difficulty][0]+thetime
                number=info[difficulty][1]+1
                info[difficulty]=[time,number]
            else:
                info[difficulty]=[thetime,1]
    
    x=[]
    for j in info:
        x.append(j)
    x.sort()
    y=[]
    for k in x:
        y.append(info[k][0]/info[k][1])
    print(x)
    fig=go.Figure(data=go.Scatter(x=x,y=y))
    fig.update_layout(
    title="Hill Climbing",
    xaxis_title="Difficulty (aka eval function)",
    yaxis_title="Execution Time (s)")
    fig.show()


def Astarplotsize():
    timearray=[]
    sizearray=[5,7,9,11]
    iter=50
    for k in sizearray:
        game_array=[]
        for i in range(iter):
            gamelen=k
            game_array.append(gamelen)
      
        thetime=0

        for j in range(iter):  
    

            thetime=Astar(gamelen).time+thetime

        

        thetime=thetime/iter
        timearray.append(thetime)
    print(timearray)
    fig=go.Figure(data=go.Scatter(x=sizearray,y=timearray))
    fig.update_layout(
    title="A*",
    xaxis_title="Size n (nxn)",
    yaxis_title="Execution Time (s)")
    fig.show()

def Astarplotdiff():
    info={}
    sizearray=[5,7,9,11]
    for h in sizearray:
        for i in range(1000):

        
            difficulty=Astar(h).eval
            thetime=Astar(h).time
            

            if difficulty in info:
                time=info[difficulty][0]+thetime
                number=info[difficulty][1]+1
                info[difficulty]=[time,number]
            else:
                info[difficulty]=[thetime,1]
    
    x=[]
    for j in info:
        x.append(j)
    x.sort()
    y=[]
    for k in x:
        y.append(info[k][0]/info[k][1])
    print(x)
    fig=go.Figure(data=go.Scatter(x=x,y=y))
    fig.update_layout(
    title="A*",
    xaxis_title="Difficulty (aka eval function)",
    yaxis_title="Execution Time (s)")
    fig.show()


def popplot():
    info={}
    sizearray=[5,7,9,11]
    for h in sizearray:
        for i in range(100):

            starttime = timeit.default_timer()
            game=geneticAlgo(50,h)
            thetime=timeit.default_timer()-starttime

            difficulty=game.eval
           
            isin=0
            for i in info:
                if abs(thetime-i)/i<.05:
                    comptime=i
                    #print(info[comptime])
                    number=info[comptime][1]+1
                    newdifficulty=info[comptime][0]+difficulty
                    info[comptime]=[newdifficulty,number]
                    isin=1
                    
        
            if isin==0:
                info[thetime]=[difficulty,1]
                

    
    x=[]
    for j in info:
        x.append(j)
    x.sort()
    y=[]
    for k in x:
        #print(info[k][0]/info[k][1])
        y.append(info[k][0]/info[k][1])
    #print(info)
    fig=go.Figure(data=go.Scatter(x=x,y=y))
    fig.update_layout(
    title="Population",
    xaxis_title="Computation Time (s)",
    yaxis_title="Evaluation Function")
    fig.show()


def popplotsize():
    timearray=[]
    sizearray=[5,7,9,11]
    iter=50
    for k in sizearray:
        game_array=[]
        for i in range(iter):
            gamelen=k
            game_array.append(gamelen)
      
        thetime=0

        for j in range(iter):  
    

            thetime=geneticAlgo(50,gamelen).time+thetime

        

        thetime=thetime/iter
        timearray.append(thetime)
    print(timearray)
    fig=go.Figure(data=go.Scatter(x=sizearray,y=timearray))
    fig.update_layout(
    title="Population",
    xaxis_title="Size n (nxn)",
    yaxis_title="Execution Time (s)")
    fig.show()

def popplotdiff():
    info={}
    sizearray=[5,7,9,11]
    for h in sizearray:
        for i in range(5):

        
            difficulty=geneticAlgo(50,h).eval
            thetime=geneticAlgo(50,h).time
            

            if difficulty in info:
                time=info[difficulty][0]+thetime
                number=info[difficulty][1]+1
                info[difficulty]=[time,number]
            else:
                info[difficulty]=[thetime,1]
    
    x=[]
    for j in info:
        x.append(j)
    x.sort()
    y=[]
    for k in x:
        y.append(info[k][0]/info[k][1])
    print(x)
    fig=go.Figure(data=go.Scatter(x=x,y=y))
    fig.update_layout(
    title="Population",
    xaxis_title="Difficulty (aka eval function)",
    yaxis_title="Execution Time (s)")
    fig.show()


def evaluation(): #final eval function that shows all the plots needed for eval
    spfplotsize()
    spfplotdiff()
    hillplotsize()
    hillplotdiff()
    Astarplotsize()
    Astarplotdiff()
    popplotsize()
    popplotdiff()

            

def show_game3(genetic,hill):
    root = Tk()
    n=len(genetic.mat_of_nodes)
    for i in range(n): #Rows
        root.columnconfigure(i, weight=1, minsize=20)
        root.rowconfigure(i, weight=1, minsize=20)
        for j in range(n): #Columns
            if i==n-1 and j==n-1:
                b = Text(master=root)
                b.insert("1.0",0)
                sp=genetic.mat_of_nodes[i][j].shortest_path
                if sp==0 and (i,j)!=(0,0):
                    sp="X"
                b.insert("2.0","\nSPath={}\nEval ={}\n#of iter={}".format(sp,genetic.eval,genetic.iter))
                b.grid(row=i, column=j)
                for j in range(0,n):
                    b = Text(master=root,height=1)
                    b.insert("1.0","")
                    b.grid(row=n+1, column=j)

            else:
                val= genetic.mat_of_nodes[i][j].value
                b = Text(master=root)
                b.insert("1.0",val)
                sp=genetic.mat_of_nodes[i][j].shortest_path
                if sp==0 and (i,j)!=(0,0):
                    sp="X"
                b.insert("2.0","\nSPath={}\n".format(sp))
                b.grid(row=i, column=j)
    

    for i in range((n+2),(2*n)+2):
        root.rowconfigure(i, weight=1, minsize=20)
        for j in range(n):
            if i==(2*n)+1 and j==n-1:
                c = Text(master=root)
                c.insert("1.0",0)
                sp=hill.mat_of_nodes[i-n-2][j].shortest_path
                if sp==0 and (i,j)!=(n+2,0):
                    sp="X"
                c.insert("2.0","\nSPath={}\nEval={}\n#of iter={}".format(sp,hill.eval,hill.iter))
                c.grid(row=i, column=j)
            else:
                val= hill.mat_of_nodes[i-n-2][j].value
                c = Text(master=root)
                c.insert("1.0",val)
                sp=hill.mat_of_nodes[i-n-2][j].shortest_path
                if sp==0 and (i,j)!=(n+2,0):
                    sp="X"
                c.insert("2.0","\nSPath={}\n".format(sp))
                c.grid(row=i, column=j)
    
    mainloop()

def show_game4(game_nodes):
    root = Tk()
    n=len(game_nodes.mat_of_nodes)
    for i in range(n): #Rows
        root.columnconfigure(i, weight=1, minsize=50)
        root.rowconfigure(i, weight=1, minsize=50)
        for j in range(n): #Columns
            if i==n-1 and j==n-1:
                b = Text(master=root)
                b.insert("1.0",0)
                sp=game_nodes.mat_of_nodes[i][j].shortest_path
                if sp==0 and (i,j)!=(0,0):
                    sp="X"
                b.grid(row=i, column=j)
            else:
                val= game_nodes.mat_of_nodes[i][j].value
                b = Text(master=root)
                b.insert("1.0",val)
                sp=game_nodes.mat_of_nodes[i][j].shortest_path
                if sp==0 and (i,j)!=(0,0):
                    sp="X"
                b.grid(row=i, column=j)

    mainloop()






popplot()
#hill_climbing(50,11)
#Astar(11)
#x=create_game(11)
#breadth_first(x)
#show_game(x)
#print(Astar(5).eval)
#hill_climbing(game_nodes,0)
#x=create_game(7)
#test()
#hillplot(11)
#show_game2(hill_climbing(50,11))
#show_game(x)
#spfplotsize()
#spfplotdiff()
#popplot()
#evaluation()
#hillplotsize()
#hillplotdiff()
#Astarplotsize()
#Astarplotdiff()
#geneticAlgo(50,5)
#popplotdiff()
#compare()
