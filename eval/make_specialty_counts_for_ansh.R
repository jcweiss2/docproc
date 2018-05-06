library(dplyr)

# Load providers data and make specialties data.
if (!("specialties_tb" %in% ls())) {
  cat("Loading providers_tb and making specialties_tb.\n")
  providers_tb = readRDS("../data/Medicare_Provider_Utilization_and_Payment_Data__Part_D_Prescriber_Summary_Table_CY2015.rds")
  specialties_tb = providers_tb %>% select(npi, specialty_description) %>%
                                    mutate(specialty_id = as.integer(specialty_description))
}

# Load NPI-to-HCPCS table.
if (!("npi_to_hcpcs_tb" %in% ls())) {
  cat("Loading npi_to_hcpcs_tb\n")
  npi_to_hcpcs_tb = read.csv(pipe("unpigz --stdout -p 12 ../data/npi_to_hcpcs_code.csv.gz"), stringsAsFactors=FALSE) %>% tbl_df
  #npi_to_hcpcs_tb = npi_to_hcpcs_tb %>% mutate(cluster = as.integer(cluster))
  npi_to_hcpcs_tb = npi_to_hcpcs_tb %>% rename(npi = National.Provider.Identifier)
}

# Make counts of HCPCS codes in each specialty.
merged_tb = specialties_tb %>% left_join(npi_to_hcpcs_tb, by="npi")
counts_tb = merged_tb %>% group_by(specialty_id, HCPCS.Code, HCPCS.Description) %>%
                          summarise(num_providers = n(),
                                    sum_counts = sum(Number.of.Services),
                                    sum_log_counts = sum(log(Number.of.Services)))
write.csv(counts_tb, file="../data/specialty_to_hcpcs_counts.csv", row.names=FALSE)
