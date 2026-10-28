########## This script is to filter, normalize and impute the MS protein abundance for a certain condition

import pandas as pd 
import numpy as np
from collections import Counter,defaultdict
from sklearn.impute import KNNImputer
import argparse
from scipy import stats
from itertools import combinations
from tqdm.auto import tqdm  
from Bio import SeqIO


parser = argparse.ArgumentParser()
parser.add_argument("Table1_Path", help="MS intensities in condition 1")
parser.add_argument("Table2_Path", help="MS intensities in condition 2")
parser.add_argument('Filer_Percent', help="Percentage of valid values")
parser.add_argument('Protein_seqs', help="Protein sequences as fasta file")
parser.add_argument("Output_filenameSequenceInfo", help="Output file name.")
parser.add_argument("Output_filenameMSFeatures", help="Output file name.")




args = parser.parse_args()

c1=pd.read_csv(args.Table1_Path) 
c2=pd.read_csv(args.Table2_Path)

thresh=float(args.Filer_Percent)

if (thresh < 0) or (thresh>1):
    raise ValueError("Error: Filter threshold should be between 0 and 1.")


data=[c1,c2]
filtered_data=[]
for d in data: ### filtering ms data based on missing values
    exps=d.iloc[:,1:]
    rownans=list(exps.isna().sum(axis=1))
    rem=[]
    for i in range(len(rownans)):
        if rownans[i]>exps.shape[1]*(1-thresh):
            rem.append(exps.index[i])
    d=d.drop(rem).reset_index(drop=True)
    filtered_data.append(d)


imputed_filtered_data=[]
for d in filtered_data: ### impute data using knn
    imputer = KNNImputer(n_neighbors=10)
    exps=d.iloc[:,1:]
    cols=list(exps.columns)
    exps=imputer.fit_transform(exps)
    exps=pd.DataFrame(exps)
    exps.columns=cols
    exps['Protein']=list(d.iloc[:,0])
    imputed_filtered_data.append(exps)



final_data=[]
for d in imputed_filtered_data: ### median normalize intensity columns
    for colIntensity in d.columns:
        if colIntensity=='Protein':
            break
        else:
            d[colIntensity]=d[colIntensity]-np.median(d[colIntensity])
    final_data.append(d)


merged_data=final_data[0].merge(final_data[1],how='inner', on='Protein')

cols=list(merged_data.columns)

newcols=[]
count=1
cond=1
for i in range(len(cols)): ### fix column names based on conditions
    c=cols[i]
    if c=='Protein':
        count=1
        cond=2
        newcols.append('Protein')
        continue

    newcols.append('Condition'+str(cond)+"_"+str(count))
    count+=1




########################## Differential Expression analysis

signifs=dict()
cols=list(merged_data.columns)
protcol=cols.index('Protein')
statistics=[]
pvals=[]
fcs1=[]
fcs2=[]
for testidx in range(merged_data.shape[0]):
    arr=stats.ttest_ind(merged_data.iloc[testidx,0:protcol].astype(float),merged_data.iloc[testidx,protcol+1:].astype(float),equal_var=False)
    statistics.append(arr.statistic)
    pvals.append(arr.pvalue)
    fcs1.append(np.mean(merged_data.iloc[testidx,0:protcol].astype(float))-(np.mean(merged_data.iloc[testidx,protcol+1:].astype(float))))
    fcs2.append((np.mean(merged_data.iloc[testidx,protcol+1:].astype(float)))-np.mean(merged_data.iloc[testidx,0:protcol].astype(float)))

merged_data['Tstat']=abs(np.array(statistics))
merged_data['Tpval']=pvals
merged_data['FC_Max']=np.maximum(np.array(fcs1),np.array(fcs2))
merged_data['FC_Min']=np.minimum(np.array(fcs1),np.array(fcs2))

adjpvals=stats.false_discovery_control(merged_data['Tpval'])
merged_data['Adjpval']=adjpvals

merged_data=merged_data.drop_duplicates().reset_index(drop=True)


fasta_dict = {record.id: str(record.seq) for record in SeqIO.parse(args.Protein_seqs, "fasta")} ## filter for proteins that have sequence information
for i in fasta_dict:
    fasta_dict[i]=fasta_dict[i].replace('*','')

merged_data=merged_data[merged_data['Protein'].isin(fasta_dict)].reset_index(drop=True)


allprots=list(set(merged_data['Protein']))

print("Merged Data Saved with "+str(len(allprots))+" similar proteins across conditions having sequence information.")
print()


interactions=list(combinations(allprots, 2))




print(len(interactions),'possible interactions will be tested.')
print()


########################### Compute features
def bicor(x, y, c=9.0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
 
    def _weighted(v):
        med = np.median(v)
        mad = np.median(np.abs(v - med))
        if mad == 0:
            return None
        u = (v - med) / (c * mad)
        w = (1 - u ** 2) ** 2
        w[np.abs(u) >= 1] = 0
        return (v - med) * w
 
    xt, yt = _weighted(x), _weighted(y)
    if xt is None or yt is None:
        return np.nan
    denom = np.sqrt(np.sum(xt ** 2)) * np.sqrt(np.sum(yt ** 2))
    return np.nan if denom == 0 else np.sum(xt * yt) / denom
 
 
def _bg_stats(corr_full):
    bg_mean, bg_std = {}, {}
    for p in corr_full.index:
        others = corr_full[p].drop(p)
        bg_mean[p] = others.mean()
        bg_std[p] = others.std(ddof=1)
    return bg_mean, bg_std
 
 
def _top_neighbors(corr_full, top_k):
    return {
        p: set(corr_full[p].drop(p).sort_values(ascending=False).head(top_k).index)
        for p in corr_full.index
    }
 
 
def _abundance_stats(wa, wb):
    """
    Level/abundance features for one pair, given two 1D replicate arrays.

    """

    e1, e2 = np.median(wa), np.median(wb)
    mad1 = np.median(np.abs(wa - e1)) * 1.4826
    mad2 = np.median(np.abs(wb - e2)) * 1.4826
    s1, s2 = mad1, mad2
 
    mean_abund = (e1 + e2) / 2
    min_abund = min(e1, e2)
    max_abund = max(e1, e2)
    exp_diff = abs(e1 - e2)
    std_diff = abs(s1 - s2)
    cosine = np.dot(wa, wb) / (np.linalg.norm(wa) * np.linalg.norm(wb))
    return mean_abund, min_abund, max_abund, exp_diff, std_diff, cosine
 
 
def _robust_zscore_rows(X):
    """
    Row-wise (per-protein) robust z-score using median/MAD*1.4826 as the
    center/scale, computed independently within the columns passed in. 
    Each condition's block is centered AND scaled by its OWN median/MAD 
    before pooling, so neither a baseline shift nor a variance difference 
    between lines survives into the pooled vector.
    """
    arr = X.to_numpy(dtype=float)
    med = np.median(arr, axis=1, keepdims=True)
    mad = np.median(np.abs(arr - med), axis=1, keepdims=True) * 1.4826
    with np.errstate(invalid='ignore', divide='ignore'):
        z = np.where(mad > 0, (arr - med) / mad, 0.0)
    return pd.DataFrame(z, index=X.index, columns=X.columns)
 
 


def build_features(data, signifs):
    """
    Compute protein interactions features between conditions
    """
    seepair=signifs
    seepair=seepair.drop_duplicates(subset=['Protein']).reset_index(drop=True)
    explocs = dict()
    count = 0
    for i in range(seepair.shape[0]):
        p = seepair.loc[i, 'Protein']
        explocs[p] = count
        count += 1

    protein_features = seepair[['Protein', 'Tstat', 'Tpval', 'FC_Max', 'FC_Min', 'Adjpval']]
    protcol=list(seepair.columns).index('Protein')
    X = seepair.set_index(seepair['Protein'])
    X = X.drop(['Tstat', 'Tpval', 'Adjpval', 'Protein', 'FC_Max', 'FC_Min'], axis=1)

    
    inter = data


    
    X1 = X.iloc[:, :protcol]   # condition 1 replicates
    X2 = X.iloc[:, protcol:]  # condition 2 replicates

    
    print('Computing correlation matrices and neighbor structures...')
    corr_full_c = X.T.corr(method='spearman')

    Z1 = _robust_zscore_rows(X1)
    Z2 = _robust_zscore_rows(X2)
    Z_combined = pd.concat([Z1, Z2], axis=1)
    corr_full_z = Z_combined.T.corr(method='pearson')

    bg_mean_c, bg_std_c = _bg_stats(corr_full_c)
    bg_mean_z, bg_std_z = _bg_stats(corr_full_z)

    TOP_K = int(0.25 * X.shape[0])
    top_neighbors_c = _top_neighbors(corr_full_c, TOP_K)
    top_neighbors_z = _top_neighbors(corr_full_z, TOP_K)

    feature_rows = []

    # Main loop: one iteration per protein interaction.
    for i in tqdm(inter, total=len(inter), desc='Pair features', unit='pair'):
        p1 = i[0]
        p2 = i[1]

        wa_c = np.asarray(X.iloc[explocs[p1], :]).ravel()
        wb_c = np.asarray(X.iloc[explocs[p2], :]).ravel()
        wa_z = np.asarray(Z_combined.iloc[explocs[p1], :]).ravel()
        wb_z = np.asarray(Z_combined.iloc[explocs[p2], :]).ravel()

        if wa_c.std() == 0 or wb_c.std() == 0:
            continue  # degenerate: constant pooled values break correlation

        # --- correlation family: combined + z-scored ---
        pear_c = stats.pearsonr(wa_c, wb_c).statistic
        spear_c = stats.spearmanr(wa_c, wb_c).statistic
        bicor_c = bicor(wa_c, wb_c)

        if wa_z.std() > 0 and wb_z.std() > 0:
            pear_z = stats.pearsonr(wa_z, wb_z).statistic
            spear_z = stats.spearmanr(wa_z, wb_z).statistic
            bicor_z = bicor(wa_z, wb_z)
        else:
            pear_z = spear_z = bicor_z = np.nan

        # --- specificity: combined (Spearman) + z-scored (Pearson) ---
        zc1 = (spear_c - bg_mean_c[p1]) / bg_std_c[p1] if bg_std_c[p1] else np.nan
        zc2 = (spear_c - bg_mean_c[p2]) / bg_std_c[p2] if bg_std_c[p2] else np.nan
        specificity_z_c = np.nanmin([zc1, zc2])

        zz1 = (pear_z - bg_mean_z[p1]) / bg_std_z[p1] if bg_std_z[p1] else np.nan
        zz2 = (pear_z - bg_mean_z[p2]) / bg_std_z[p2] if bg_std_z[p2] else np.nan
        specificity_z_z = np.nanmin([zz1, zz2])

        # --- topology: combined + z-scored ---
        recip_c = int((p2 in top_neighbors_c[p1]) and (p1 in top_neighbors_c[p2]))
        jden_c = len(top_neighbors_c[p1] | top_neighbors_c[p2])
        jacc_c = len(top_neighbors_c[p1] & top_neighbors_c[p2]) / jden_c if jden_c else 0.0

        recip_z = int((p2 in top_neighbors_z[p1]) and (p1 in top_neighbors_z[p2]))
        jden_z = len(top_neighbors_z[p1] | top_neighbors_z[p2])
        jacc_z = len(top_neighbors_z[p1] & top_neighbors_z[p2]) / jden_z if jden_z else 0.0

        # --- level/abundance: combined only (robust median/MAD) ---
        (mean_abund_c, min_abund_c, max_abund_c, exp_diff_c, std_diff_c,
            cosine_c) = _abundance_stats(wa_c, wb_c)

        feature_rows.append([
            tuple(sorted((p1, p2))),
            pear_c, spear_c, bicor_c, specificity_z_c,
            recip_c, jacc_c,
            mean_abund_c, min_abund_c, max_abund_c, exp_diff_c, std_diff_c,
            cosine_c,
            pear_z, spear_z, bicor_z, specificity_z_z,
            recip_z, jacc_z,
        ])

    see = pd.DataFrame(feature_rows, columns=[
        'Pair',
        'Pearson_Combined', 'Spearman_Combined', 'Bicor_Combined', 'SpecificityZ_Combined',
        'ReciprocalTopK_Combined', 'JaccardNeighbors_Combined',
        'MeanAbund_Combined', 'MinAbund_Combined', 'MaxAbund_Combined',
        'ExpDiff_Combined', 'StdDiff_Combined', 'Cosine_Combined',
        'Pearson_Zscore', 'Spearman_Zscore', 'Bicor_Zscore', 'SpecificityZ_Zscore',
        'ReciprocalTopK_Zscore', 'JaccardNeighbors_Zscore',
    ])
    

    addedfeats = []
    for pair in tqdm(see['Pair'], total=len(see), desc='Protein-level features', unit='pair'):
        p1 = pair[0]
        p2 = pair[1]
        feats1 = np.array(protein_features.iloc[explocs[p1], 1:])
        feats2 = np.array(protein_features.iloc[explocs[p2], 1:])

        maxfeats = list(np.maximum(feats1, feats2))
        minfeats = list(np.minimum(feats1, feats2))

        arr = []
        arr.extend(maxfeats)
        arr.extend(minfeats)

        addedfeats.append(arr)
    addedfeats = pd.DataFrame(addedfeats)
    addedfeats.columns = ['MaxTstat', 'MaxTpval', 'MaxFC_Max', 'MaxFC_Min', 'MaxAdjpval',
                            'MinTstat', 'MinTpval', 'MinFC_Max', 'MinFC_Min', 'MinAdjpval']
    finalinfo = pd.concat([see, addedfeats], axis=1)
 
    return finalinfo
        

data=build_features(interactions,merged_data)






inters=[]
for i in data['Pair']:
    if (i[0] in fasta_dict) and (i[1] in fasta_dict):
        inters.append([i[0],i[1],fasta_dict[i[0]],fasta_dict[i[1]]])

inters=pd.DataFrame(inters)
inters.columns=['Protein1','Protein2','Seq1','Seq2']
inters.to_csv(args.Output_filenameSequenceInfo,index=False)


data.to_csv(args.Output_filenameMSFeatures,index=False)
