library(dplyr)
library(reshape2)

if (1) {

# Load clusters.
if (!("clusters_tb" %in% ls())) {
  cat("Loading clusters_tb\n")
  clusters_tb = read.csv(pipe("unpigz --stdout -p 12 ../data/180426clusters60merged.csv.gz"), stringsAsFactors=FALSE) %>% tbl_df
  clusters_tb = clusters_tb %>% select(-X) %>% mutate(cluster = as.integer(cluster))
}

# Load kmeans clusters.
if (!("kmeans_tb" %in% ls())) {
  cat("Loading kmeans_tb\n")
  kmeans_tb = read.csv(pipe("unpigz --stdout -p 12 ../data/clusters_kmeans_procs_2018-04-26.gz"), stringsAsFactors=FALSE) %>% tbl_df
  kmeans_tb = kmeans_tb %>% select(-X) %>% mutate(kmeans_cluster = as.integer(cluster)) %>% select(-cluster)
}

# Load providers data and make specialties data.
if (!("specialties_tb" %in% ls())) {
  cat("Loading providers_tb and making specialties_tb.\n")
  providers_tb = readRDS("../data/Medicare_Provider_Utilization_and_Payment_Data__Part_D_Prescriber_Summary_Table_CY2015.rds")
  specialties_tb = providers_tb %>% select(npi, specialty_description) %>%
                                    mutate(specialty_id = as.integer(specialty_description))
}

# Make both_tb.
cat("Make both_tb.\n")
both_tb = clusters_tb %>% full_join(kmeans_tb, by="npi") %>%
                          full_join(specialties_tb, by="npi")

# Load NPI-to-HCPCS table.
if (!("npi_to_hcpcs_tb" %in% ls())) {
  cat("Loading npi_to_hcpcs_tb\n")
  npi_to_hcpcs_tb = read.csv(pipe("unpigz --stdout -p 12 ../data/npi_to_hcpcs_code.csv.gz"), stringsAsFactors=FALSE) %>% tbl_df
  #npi_to_hcpcs_tb = npi_to_hcpcs_tb %>% mutate(cluster = as.integer(cluster))
  npi_to_hcpcs_tb = npi_to_hcpcs_tb %>% rename(npi = National.Provider.Identifier)
}

### Make per-cluster counts.

# Make counts of HCPCS codes in each cluster.
merged_tb = clusters_tb %>% left_join(npi_to_hcpcs_tb, by="npi")
hcpcs_counts_tb = merged_tb %>% group_by(cluster, HCPCS.Code, HCPCS.Description) %>%
                                summarise(num_providers = n(),
                                          sum_counts = sum(Number.of.Services),
                                          sum_log_counts = sum(log(Number.of.Services)))

# Make counts of specialties in each cluster.
cluster_to_specialty_counts_tb = both_tb %>% group_by(cluster, specialty_description, specialty_id) %>%
                                             summarise(num_providers = n())
cluster_to_specialty_counts_tb = cluster_to_specialty_counts_tb %>% filter(!is.na(specialty_id))

# Make counts of specialties in each cluster.
kmeans_to_specialty_counts_tb = both_tb %>% group_by(kmeans_cluster, specialty_description, specialty_id) %>%
                                summarise(num_providers = n())
kmeans_to_specialty_counts_tb = kmeans_to_specialty_counts_tb %>% filter(!is.na(specialty_id))
kmeans_to_specialty_counts_tb = kmeans_to_specialty_counts_tb %>% rename(cluster = kmeans_cluster)

}

### Make grids.

make_grid = function(counts_tb, n, value.var) {

  #counts_tb = specialty_counts_tb
  #n = 5
  #value.var = "specialty_description"

  # Rank HCPCS codes within clusters.
  #ranked_counts_tb = counts_tb %>% arrange(-sum_log_counts) %>% group_by(cluster) %>% mutate(rank = row_number()) %>% ungroup
  ranked_counts_tb = counts_tb %>% arrange(-num_providers) %>% group_by(cluster) %>% mutate(rank = row_number()) %>% ungroup
  ranked_counts_tb = ranked_counts_tb %>% arrange(cluster, rank)

  # Take the top 3 HCPCS codes for each cluster.
  top3_counts_tb = ranked_counts_tb %>% group_by(cluster) %>% top_n(n, -rank) %>% ungroup

  # Make a grid, with labels and counts in separate grids.
  top3_label_grid_tb = top3_counts_tb %>% dcast(cluster ~ rank, value.var=value.var) %>% tbl_df
  top3_count_grid_tb = top3_counts_tb %>% dcast(cluster ~ rank, value.var="num_providers") %>% tbl_df

  # Combine the labels and counts into one grid.
  top3_grid_tb = top3_label_grid_tb
  cn = as.character(1:n)
  for (c in cn) {
    top3_grid_tb[[c]] = sprintf("%s (%d)", top3_label_grid_tb[[c]], top3_count_grid_tb[[c]])
  }

  # Arrange by the count of the first column.
  top3_grid_tb$top_count = top3_count_grid_tb[["1"]]
  top3_grid_tb = top3_grid_tb %>% arrange(-top_count) %>% select(-top_count)

  top3_grid_tb
}
hcpcs_grid_tb = make_grid(hcpcs_counts_tb, 3, "HCPCS.Description")
cluster_to_specialty_grid_tb = make_grid(cluster_to_specialty_counts_tb, 3, "specialty_description")
kmeans_to_specialty_grid_tb = make_grid(kmeans_to_specialty_counts_tb, 3, "specialty_description")

cluster_to_specialty_grid_tb = cluster_to_specialty_grid_tb %>% filter(!is.na(cluster))
kmeans_to_specialty_grid_tb = kmeans_to_specialty_grid_tb %>% filter(!is.na(cluster))

### Print grids.
print_grid = function(grid_tb) {
  #grid_tb = cluster_to_specialty_grid_tb
  cat("\\begin{tabular}{l|lllll}\n")
  cat("\\hline\n")
  #cat("cluster & First & Second & Third & Fourth & Fifth \\\\ \n")
  cat("Cluster & Most Common Specialty (Count) & Second Most Common Specialty (Count) & Third Most Common Specialty (Count) \\\\ \n")
  cat("\\hline\n")
  for (i in 1:nrow(grid_tb)) {
    row = grid_tb[i,]
    cat(sprintf("%s", row$cluster))
    for (i in 1:3) {
      if (row[i+1] == "NA (NA)") {
        cat(" & -")
      } else {
        cat(sprintf(" & %s", row[i+1]))
      }
    }
    cat("\\\\ \n")
  }
  cat("\\hline\n")
  cat("\\end{tabular}\n")
}
cat("\\begin{center}(a) Our Clustering\\end{center} \\\\ \n")
print_grid(cluster_to_specialty_grid_tb)
cat("\\begin{center}(b) K-means Clustering\\end{center} \\\\ \n")
print_grid(kmeans_to_specialty_grid_tb)

