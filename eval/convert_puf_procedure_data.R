library(dplyr)
filename = "../data/Medicare Provider Utilization and Payment Data: Physician and Other Supplier PUF CY2015.csv"
head_tb = read.csv(pipe(sprintf("cat \"%s\" | head -n 2", filename)))

col_classes = rep("character", length(colnames(head_tb)))
full_tb = read.csv(pipe(sprintf("cat \"%s\"", filename)), colClasses=col_classes)

# Select subset of columns.
npi_to_hcpcs_code_tb = full_tb %>% select(National.Provider.Identifier, 
                                          HCPCS.Code,
                                          HCPCS.Description,
                                          HCPCS.Drug.Indicator,
                                          Number.of.Services)

# Merge counts across rows (different places of service have different rows).
npi_to_hcpcs_tb = npi_to_hcpcs_tb %>% group_by(National.Provider.Identifier, HCPCS.Code, HCPCS.Description, HCPCS.Drug.Indicator) %>%
                                      summarise(Number.of.Services = sum(Number.of.Services)) %>%
                                      filter(National.Provider.Identifier != 1)

#saveRDS(npi_to_hcpcs_code_tb, file="../data/npi_to_hcpcs_code.rds")
write.csv(npi_to_hcpcs_code_tb, file="../data/npi_to_hcpcs_code.csv", row.names=FALSE)
system("pigz -p 12 ../data/npi_to_hcpcs_code.csv")

# npi_to_hcpcs_code_without_desc_tb = full_tb %>% select(National.Provider.Identifier, 
#                                                        HCPCS.Code,
#                                                        HCPCS.Drug.Indicator,
#                                                        Number.of.Services)
# #saveRDS(npi_to_hcpcs_code_without_desc_tb, file="../data/npi_to_hcpcs_code_without_desc.rds")
# write.csv(npi_to_hcpcs_code_without_desc_tb, file="../data/npi_to_hcpcs_code_without_desc.csv", row.names=FALSE)
# system("pigz -p 12 ../data/npi_to_hcpcs_code_without_desc.csv")

