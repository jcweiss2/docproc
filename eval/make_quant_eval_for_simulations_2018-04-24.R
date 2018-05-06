library(cluster)
library(clusterCrit)
library(dplyr)
library(maxmatching)
library(proxy)

library(foreach)
library(parallel)

options(mc.cores=16)

date_tag = "2018-04-24"
directory_tag = sprintf("simulations_%s", date_tag)

input_options_tb = read.csv(text="centroids,samples,dims,sd,explode
25,1000,1000,0.01,10000
25,1000,100000,0.01,10000
25,1000,1000,0.1,10000
25,1000,1000,0.01,1000000
100,1000,1000,0.01,10000
25,100,1000,0.01,10000", colClasses=rep("character", 5))

#dist_tags = c("cosine", "Euclidean")
dist_tags = c("cosine")
space_tags = c("hidden", "output")

mclapply(1:nrow(input_options_tb), function(row_idx) {
r = input_options_tb[row_idx,]

# Set input options.
#input_tag = "centroids25_samples100_dims1000_sd0.01_explode10000"
input_tag = sprintf("centroids%s_samples%s_dims%s_sd%s_explode%s",
                    r$centroids, r$samples, r$dims, r$sd, r$explode)

for (dist_tag in dist_tags) {
for (space_tag in space_tags) {

cat(sprintf("\n=================\n"))

# Set output options.
# dist_tag = "cosine"
# #dist_tag = "Euclidean"
# space_tag = "hidden"
# #space_tag = "output"
output_tag = sprintf("%s_space_%s.dist_%s.date_%s", input_tag, space_tag, dist_tag, date_tag)
cat(sprintf("Running with output_tag %s\n", output_tag))

# Load input.
read_tb = function(filename) {
  full_filename = sprintf("../data/%s/%s", directory_tag, filename)
  cat(sprintf("Loading %s\n", full_filename))
  read.csv(full_filename, stringsAsFactors=FALSE) %>% select(-X) %>% tbl_df
}
clusters_hidden_kmeans = read_tb(sprintf("%s_clusters_hidden_kmeans.csv", input_tag))
clusters_kmeans = read_tb(sprintf("%s_clusters_kmeans.csv", input_tag))
clusters_ours = read_tb(sprintf("%s_clusters_ours.csv", input_tag))
clusters_ours_merged = read_tb(sprintf("%s_clusters_ours_merged.csv", input_tag))
hidden_tb = read_tb(sprintf("%s_hidden.csv", input_tag))
original_tb = read_tb(sprintf("%s_original.csv", input_tag))
summary_tb = read_tb(sprintf("%s_summary.csv", input_tag))

# print("clusters_hidden_kmeans"); print(clusters_hidden_kmeans, n=3)
# print("clusters_kmeans"); print(clusters_kmeans, n=3)
# print("clusters_ours"); print(clusters_ours, n=3)
# print("clusters_ours_merged"); print(clusters_ours_merged, n=3)
# print("hidden_tb"); print(hidden_tb, n=3)
# print("original_tb"); print(original_tb, n=3)
# print("summary_tb"); print(summary_tb, n=3)

# Determine output locations.
output_dir = sprintf("../results/%s", directory_tag)
system(sprintf("mkdir -p %s", output_dir))
silhouette_filename = sprintf("%s/silhouette.%s.pdf", output_dir, output_tag)
ext_tb_filename = sprintf("%s/external_criteria.%s.csv", output_dir, output_tag)
int_tb_filename = sprintf("%s/internal_criteria.%s.csv", output_dir, output_tag)
best_int_criteria_filename = sprintf("%s/best_int_criteria.%s.csv", output_dir, output_tag)

# Determine true cluster assignments.
truth_tb = clusters_ours$truth
stopifnot(truth_tb == original_tb$truth)
stopifnot(truth_tb == clusters_hidden_kmeans$truth)
stopifnot(truth_tb == clusters_kmeans$truth)
stopifnot(truth_tb == clusters_ours$truth)
stopifnot(truth_tb == clusters_ours_merged$truth)
stopifnot(truth_tb == hidden_tb$truth)
stopifnot(truth_tb == original_tb$truth)

# Make matrices.
original_x = as.matrix(original_tb %>% select(-truth))
hidden_x = as.matrix(hidden_tb %>% select(-truth))

# # Match cluster labels to ground truth.
# weighted_edge_tb = clusters_ours %>% mutate(cluster = sprintf("cluster_%d", cluster),
#                                             truth = sprintf("truth_%d", truth)) %>%
#                                      group_by(cluster, truth) %>%
#                                      summarise(n = n())
# weights = weighted_edge_tb$n
# edges_tb = weighted_edge_tb %>% select(cluster, truth)
# graph = igraph::make_undirected_graph(as.vector(t(as.matrix(edges_tb))))
# #names(weights) = igraph::E(graph)
# igraph::E(graph)$weight = weights
# matching_obj = maxmatching(graph, weighted=TRUE)
# #matching_vec = matching_obj$matching[seq(1,length(matching_obj$matching),by=2)]
# #matching_tb = data_frame(cluster=names(matching_vec), truth=matching_vec)

# Combine all clustering objects.
all_tb = bind_cols(clusters_ours           %>% transmute(truth         = sprintf("truth_%d", truth)),
                   clusters_hidden_kmeans  %>% transmute(hidden_kmeans = sprintf("hidden_kmeans_%d", cluster)),
                   clusters_kmeans         %>% transmute(kmeans        = sprintf("kmeans_%d", cluster)),
                   clusters_ours           %>% transmute(ours          = sprintf("ours_%d", cluster)),
                   clusters_ours_merged    %>% transmute(ours_merged   = sprintf("ours_merged_%d", cluster)))
clustering_names = c("truth",
                     "hidden_kmeans",
                     "kmeans",
                     "ours",
                     "ours_merged")

# Choose covariates for the space we are in.
cat("Choose covariates.\n")
if (space_tag == "hidden") {
  space_x = hidden_x
} else if (space_tag == "output") {
  space_x = original_x
} else {
  stop(sprintf("Unknown space_tag: %s", space_tag))
}

# Compute distance.
cat("Compute distance.\n")
dist_mat = dist(space_x, space_x, method=dist_tag)

# Make silhoutte plot.
cat("Make silhoutte plot.\n")
pdf(silhouette_filename)
for (n in clustering_names) {
  cluster_idxs = as.integer(as.factor(all_tb[[n]]))
  sil = silhouette(cluster_idxs, dist_mat)
  plot(sil, main=sprintf("silhouette plot for %s", n))
}
dev.off()

# Carry out external comparisons.
cat("Carry out external comparisons.\n")
ext_tb = lapply(clustering_names, function(n) {
  cluster_idxs = as.integer(as.factor(all_tb[[n]]))
  truth_idxs = as.integer(as.factor(all_tb[["truth"]]))
  extCriteria(cluster_idxs, truth_idxs, "all") %>% as_data_frame %>% mutate(clustering=n)
}) %>% bind_rows %>% select(clustering, everything())
write.csv(ext_tb, file=ext_tb_filename, row.names=FALSE)

# Carry out internal comparisons.
cat("Carry out internal comparisons.\n")
int_tb = lapply(clustering_names, function(n) {
  cluster_idxs = as.integer(as.factor(all_tb[[n]]))
  intCriteria(space_x, cluster_idxs, "all") %>% as_data_frame %>% mutate(clustering=n)
}) %>% bind_rows %>% select(clustering, everything())
write.csv(int_tb, file=int_tb_filename, row.names=FALSE)

}
}
})
