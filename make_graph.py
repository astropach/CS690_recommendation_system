import json
import random
import numpy as np
import os
import snap
# from tqdm import tqdm
import torch
from torch_geometric.data import Data
random.seed(1000)

import pandas as pd

save_dir = '.' 
K = 10

df_inter = pd.read_csv('filtered_goodreads.csv',index_col=0)
df_inter=df_inter[df_inter.rating>3.5]
df_inter = df_inter[df_inter.user_id.isin(df_inter.groupby('user_id').count().reset_index().sort_values('user_id',ascending=False)[:100000].user_id.values)]

df_inter = df_inter[df_inter.book_id_orig.isin(df_inter.groupby('book_id_orig').count().reset_index().sort_values('book_id',ascending=False)[:10000].book_id_orig.values)]


df_inter = df_inter[df_inter.user_id.isin(random.sample(list(df_inter.user_id.unique()),30000))]
print(df_inter.head())
# df_inter.to_csv('finaldata.csv')

G = snap.TUNGraph().New()


curUserIdx=0

userToId = {}

for rw in zip(df_inter['user_id']):
    if rw[0] not in userToId:
        G.AddNode(curUserIdx)
        userToId[rw[0]]=curUserIdx
        curUserIdx+=1

maxUserPid = max([x.GetId() for x in G.Nodes()])     # will start book IDs after this ID
assert maxUserPid == len([x for x in G.Nodes()]) - 1
print(maxUserPid)


# bookToId={}
# edge_weights=[]
all_edges=[]
currBookIdx = maxUserPid + 1 # start books idxs here. (see above for explanation)
# userInfo = {} # maps the user ID (pid) to information about that user
bookToId = {} # maps the book URI to its new index (which we are generating, unlike the pid above) and other info about the book
# Note: some users have same book multiple times. I will ignore those and just add 1 edge
for rw in zip(df_inter['user_id'],df_inter['book_id_orig'],df_inter['rating']):

        # First time seeing this book. Add a new node to the graph
#     if rw[2]>3.5:
    if rw[1] not in bookToId:
        bookToId[rw[1]] = currBookIdx#{'bookid': currbookIdx, 'track_name': track_name, 'artist_name': artist_name,
                           #    'artist_uri': artist_uri}
        assert not G.IsNode(currBookIdx)
        G.AddNode(currBookIdx)
        currBookIdx += 1
    # Add edge between the current user and book

    G.AddEdge(userToId[rw[0]], bookToId[rw[1]])
    assert userToId[rw[0]]< bookToId[rw[1]]
#     all_edges.append([userToId[rw[0]], bookToId[rw[1]]])
#     all_edges.append([ bookToId[rw[1]], userToId[rw[0]]])
#     edge_weights.append(rw[2])
num_usr_orig = len([x for x in G.Nodes() if x.GetId() <= maxUserPid])
num_book_orig = len([x for x in G.Nodes() if x.GetId() > maxUserPid])
print("Original graph:")
print(f"Num nodes: {len([x for x in G.Nodes()])} ({num_usr_orig} users, {num_book_orig} unique books)")
print(f"Num edges: {len([x for x in G.Edges()])} (undirected)")
# print(len(edge_weights))
  


# Get K-Core subgraph
kcore = G.GetKCore(K)
if kcore.Empty():
    raise Exception(f"No Core exists for K={K}")

# Print the same stats as above, but after calculating K-core subgraph
num_usr_kcore = len([x for x in kcore.Nodes() if x.GetId() <= maxUserPid])
num_book_kcore = len([x for x in kcore.Nodes() if x.GetId() > maxUserPid])
print(f"K-core graph with K={K}:")
print(f"Num nodes: {len([x for x in kcore.Nodes()])} ({num_usr_kcore} playlists, {num_book_kcore} unique songs)")
kcore_num_edges = len([x for x in kcore.Edges()])
print(f"Num edges: {kcore_num_edges} (undirected)")


# Need to re-index new graph to have nodes in continuous sequence. After finding the K-core, we will have lost a lot of
# nodes, so indices will no longer be from 0...num_nodes. That will cause some issues later in PyG if we don't fix it here.
cnt = 0
oldToNewId_user = {}
oldToNewId_book = {}
for NI in kcore.Nodes(): # will be in sorted order already
    old_id = NI.GetId()
    assert old_id not in oldToNewId_book and old_id not in oldToNewId_user # each should only appear once
    new_id = cnt
    if old_id <= num_usr_orig - 1:
        oldToNewId_user[old_id] = new_id
    else:
        oldToNewId_book[old_id] = new_id
    cnt += 1

# A few error checks. Nodes in the for loop above should be in sorted order, so all playlists will still be before all songs.
assert max(oldToNewId_user.values()) == num_usr_kcore-1
assert len(oldToNewId_user.values()) == num_usr_kcore
assert max(oldToNewId_book.values()) == len([x for x in kcore.Nodes()]) - 1
assert len(oldToNewId_book.values()) == num_book_kcore


# Just rearranging the info saved above to get useful information about songs and playlists in our K-Core graph.
# These will only be used for analyzing our results after training the model
bookInfo = {} # will map new song index/ID -> a dictionary containing some information about that song
for track_uri, info in bookToId.items():
    if info in oldToNewId_book: # only keeping songs that ended up in the K-Core graph
        new_id = oldToNewId_book[info]
        bookInfo[new_id] = track_uri#{'track_uri': track_uri, 'track_name': info['track_name'], 'artist_uri': info['artist_uri'],
                            #'artist_name': info['artist_name']}
userInfo = {k: oldToNewId_user[v] for k,v in userToId.items() if v in oldToNewId_user}
# print(userInfo)




for EI in (kcore.Edges()):
    edge_info = [oldToNewId_user[EI.GetSrcNId()], oldToNewId_book[EI.GetDstNId()]] # using new node IDs instead of old ones
    all_edges.append(edge_info)
#     all_edges.append(edge_info[::-1]) 
# print(userToId)



    
    

edge_idx = torch.LongTensor(all_edges)
# pt = edge_idx.t().contiguous()
# print(pt[:,pt[0,:]<pt[1,:]])

data = Data(edge_index = edge_idx.t().contiguous(), num_nodes=kcore.GetNodes())


# Save Data object (for training model), some dataset stats/metadata, and song/playlist info (used for post-training analysis)
torch.save(data, os.path.join(save_dir, 'data_object.pt'))
stats = {'num_playlists': num_usr_kcore, 'num_nodes': num_usr_kcore+num_book_kcore, 'num_edges_undirected': len([x for x in kcore.Edges()])}
with open(os.path.join(save_dir, 'dataset_stats.json'), 'w') as f:
    json.dump(stats, f)
with open(os.path.join(save_dir, 'user_info.json'), 'w') as f:
    json.dump(userInfo, f)
with open(os.path.join(save_dir, 'book_info.json'), 'w') as f:
    json.dump(bookInfo, f)