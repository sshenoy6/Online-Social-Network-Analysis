"""
cluster.py
"""
import networkx as nx
import matplotlib.pyplot as plt
import collect
import pickle
import random
import operator
from collections import defaultdict

def create_graph(tweets):
    graph = nx.Graph()
    friends_of_authors = pickle.load(open("friends.pkl", "rb"))
    followers = pickle.load(open("followers.pkl", "rb"))
    followers_screen_name = [f['screen_name'] for f in followers]
    graph.add_node('JohnGrisham')
    
    tweet_authors = [t['user']['screen_name'] for t in tweets]
    
    for i in range(len(tweet_authors)):
        graph.add_node(tweet_authors[i])
        graph.add_edge('JohnGrisham',tweet_authors[i],weight=1)
        if tweet_authors[i] in friends_of_authors.keys():  
            for j in range(len(friends_of_authors[tweet_authors[i]])):
                if friends_of_authors[tweet_authors[i]][j]['screen_name'] in followers_screen_name:
                    if friends_of_authors[tweet_authors[i]][j]['screen_name'] not in graph.nodes(): 
                        graph.add_node(friends_of_authors[tweet_authors[i]][j]['screen_name'])
                        graph.add_edge(tweet_authors[i],friends_of_authors[tweet_authors[i]][j]['screen_name'],weight=1)

    return graph
    
    
def cluster_by_betweenness(graph):
    edge_centrality = nx.edge_betweenness_centrality(graph)
    
    edge_centrality = dict(sorted(edge_centrality.items(),key = operator.itemgetter(1),reverse = True))
    new_graph = graph.copy()
    components = []
    community_size = defaultdict(int)
    while len(components) <= 2:
        edge_highest_betweenness = max(edge_centrality,key = edge_centrality.get)
        #print(edge_highest_betweenness)
        new_graph.remove_edge(*edge_highest_betweenness)
        del edge_centrality[edge_highest_betweenness]
        
        components = [component for component in nx.connected_component_subgraphs(new_graph)]

    for i in range(len(components)):
        community_size[i] = len(components[i].nodes())
        
    return components,community_size

def cluster_by_min_degree(graph,min_degree):
    
    degree_nodes = graph.degree()
    nodes_to_remove = [key for key,value in degree_nodes.items() if degree_nodes[key] < min_degree]
    graph.remove_nodes_from(nodes_to_remove)
    
    return graph

def draw_network(graph, tweets, filename):
    """
    Draw the network to a file. Only label the candidate nodes; the friend
    nodes should have no labels (to reduce clutter).
    Methods you'll need include networkx.draw_networkx, plt.figure, and plt.savefig.
    Your figure does not have to look exactly the same as mine, but try to
    make it look presentable.
    """
    node_labels = {}
    source_list = [t['user']['screen_name'] for t in tweets]
    
    for node in graph.nodes():
        if str(node) in source_list:
            node_labels[node] = node
    
    plt.figure(figsize=(18,18))
    pos=nx.spring_layout(graph)
    nx.draw(graph,pos,with_labels=False,node_size=60,font_size=14,edge_color='0.50')
    nx.draw_networkx_labels(G=graph,pos=pos,labels=node_labels,font_size=14,edge_color='0.50')
    plt.savefig(filename)

def main():
    f = open('cluster_data_results.txt','w',encoding='utf-8')
    tweets = pickle.load(open("tweets.pkl", "rb"))
    graph = create_graph(tweets)
    draw_network(graph,tweets,'social_network_plot.png')
    components,community_size = cluster_by_betweenness(graph)
    number_of_communities = len(community_size.keys())
    f.write("\nNumber of communities discovered:%s"%number_of_communities)
    average_number_of_users = sum(community_size.values())/len(community_size.keys())
    f.write("\nAverage number of users per community:%s"%average_number_of_users)
    f.close()
    for i in range(len(components)):
        j = i+1
        draw_network(components[i],tweets,"communities"+str(j)+".png")
    
    
if __name__ == '__main__':
    main()
    