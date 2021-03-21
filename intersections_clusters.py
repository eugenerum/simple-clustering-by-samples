df_new = pd.DataFrame()
df_new["targets"] = df['targets']
df_new["preds"] = df['preds']

for i in range(7):
    globals()['cluster_%s' % i] = 0

for num_clusters in range(7):
    df_new1 = pd.DataFrame()
    df_new1 = df_new.loc[df['preds'] == num_clusters]
    comp = "\n".join(df_new1["targets"].unique())
    comp = comp.split("\n")
    
    for num_clusters_check in range(7):
        if num_clusters == num_clusters_check:
            continue
        else:
            df_new1 = pd.DataFrame()
            df_new1 = df_new.loc[df['preds'] == num_clusters_check]
            comp1 = "\n".join(df_new1["targets"].unique())
            comp1 = comp1.split("\n")
            
            a = comp1
            b = comp
            c = []
            for i in a:
                for j in b:
                    if i == j:
                        c.append(i)
                        break
                        
            globals()['cluster_%s' % num_clusters] = globals()['cluster_%s' % num_clusters] + len(c)
  
counts_full = []
for i in range(7):
    counts_full.append(globals()['cluster_%s' % i])
    
import matplotlib.pyplot as plt

groups = [f"cluster_{i}" for i in range(7)]
plt.subplots(figsize=(10,5))
plt.title("Feature intersection diagram")
plt.xlabel("cluster")
plt.ylabel("Count intersections")
plt.bar(groups, counts_full)
