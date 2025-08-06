#!/bin/bash

# Check if the required inputs are provided
if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 <SourceStorageAccountName> <DestinationStorageAccountName> <ContainerName> <DirectoryPath>"
    exit 1
fi

# Assign inputs to variables
source_storage_account_name="$1"
destination_storage_account_name="$2"
container_name="$3"
directory_path="$4"

# Get the source and destination storage SAS tokens
# The SAS token is valid for 5 days and has permissions to read, write, and list the container
echo "$(date) Generating SAS tokens for source and destination storage accounts..."
end=$(date -u -d "5 days" '+%Y-%m-%dT%H:%MZ')
source_sas=$(az storage container generate-sas \
    --account-name "${source_storage_account_name}" \
    --as-user \
    --auth-mode login \
    --name "${container_name}" \
    --permissions acrwl \
--expiry "$end")
destination_sas=$(az storage container generate-sas \
    --account-name "${destination_storage_account_name}" \
    --as-user \
    --auth-mode login \
    --name "${container_name}" \
    --permissions acrwl \
--expiry "$end")

source_sas=$(echo $source_sas | tr -d '"')
destination_sas=$(echo $destination_sas | tr -d '"')

# Check if SAS tokens are empty
if [ -z "$source_sas" ] || [ -z "$destination_sas" ]; then
    echo "Error: Failed to generate SAS tokens for source or destination storage account."
    exit 1
fi

# Display the generated SAS tokens for debugging purposes
# echo "Source SAS Token: $source_sas"
# echo "Destination SAS Token: $destination_sas"

sync_container_contents() {
    echo -e "\n\nSyncing contents of ${container_name}${directory_path} from https://${source_storage_account_name}.blob.core.windows.net \
    to https://${destination_storage_account_name}.blob.core.windows.net"
    
    azcopy sync \
    "https://${source_storage_account_name}.blob.core.windows.net/${container_name}${directory_path}?${source_sas}" \
    "https://${destination_storage_account_name}.blob.core.windows.net/${container_name}${directory_path}?${destination_sas}" \
    --recursive=true
    # azcopy sync \
    #     "https://${source_storage_account_name}.blob.core.windows.net/${container_name}${directory_path}?${source_sas}" \
    #     "https://${destination_storage_account_name}.blob.core.windows.net/${container_name}${directory_path}?${destination_sas}" \
    #     --recursive=true
    
    # Capture the exit status of the azcopy sync command
    sync_status=$?
    
    # Check the exit status and return appropriate output
    if [ $sync_status -eq 0 ]; then
        echo "Sync completed successfully."
    else
        echo "Sync failed with exit status $sync_status."
        exit $sync_status
    fi
}

# Function to summarize the Blob content and calculate sizes
summarize_blob_content() {
    local file_path="$1"
    
    # Initialize counters
    file_count=0
    total_size_mib=0
    total_size_gib=0
    total_size_tib=0
    
    # Read the file line by line
    while IFS= read -r line; do
        # Increment file count
        ((file_count++))
        # Print file count for every 100 multiples
        if (( file_count % 5000 == 0 )); then
            echo "Processed $file_count files so far for $file_path..."
        fi
        # Extract the content length value and unit in one pass
        if [[ $line =~ Content\ Length:\ ([0-9.]+)\ ([KMG]iB) ]]; then
            content_length=${BASH_REMATCH[1]}
            unit=${BASH_REMATCH[2]}
            
            # Convert size to MiB using a lookup table
            case "$unit" in
                KiB) size_mib=$(awk "BEGIN {print $content_length / 1024}") ;;
                MiB) size_mib=$content_length ;;
                GiB) size_mib=$(awk "BEGIN {print $content_length * 1024}") ;;
                *) size_mib=0 ;; # Default to 0 if unit is unrecognized
            esac
            
            # Add to total size
            total_size_mib=$(awk "BEGIN {print $total_size_mib + $size_mib}")
        fi
    done < "$file_path"
    
    total_size_gib=$(awk "BEGIN {print $total_size_mib / 1024}")
    total_size_tib=$(awk "BEGIN {print $total_size_mib / 1048576}")
    
    # Output results
    printf "File: %s | Files: %d | Total Size: %.2f MiB, %.2f GiB, %.2f TiB\n" \
    "$file_path" "$file_count" "$total_size_mib" "$total_size_gib" "$total_size_tib"
}

# Fetch the contents of the source and destination containers
# The list command will output the contents to a file named <StorageAccountName>_<ContainerName>_<DirectoryPath>_contents
fetch_container_contents() {
    echo "Fetching contents of the source container: ${source_storage_account_name}/${container_name}${directory_path}..."
    {
        azcopy list \
        "https://${source_storage_account_name}.blob.core.windows.net/${container_name}${directory_path}?${source_sas}" \
        > "${source_storage_account_name}_${container_name}_${directory_path}_contents" &
    }
    
    echo "Fetching contents of the destination container: ${destination_storage_account_name}/${container_name}${directory_path}..."
    {
        azcopy list \
        "https://${destination_storage_account_name}.blob.core.windows.net/${container_name}${directory_path}?${destination_sas}" \
        > "${destination_storage_account_name}_${container_name}_${directory_path}_contents" &
    }
    
    # Wait for both background processes to complete
    wait
    echo "Contents fetched successfully."
}

summarize_blobs() {
    echo "Summarizing blobs in the source container..."
    summarize_blob_content "${source_storage_account_name}_${container_name}_${directory_path}_contents" &
    
    echo "Summarizing blobs in the destination container..."
    summarize_blob_content "${destination_storage_account_name}_${container_name}_${directory_path}_contents" &
    
    # Wait for both background processes to complete
    wait
    echo "Blob summaries completed."
}

diff_between_containers() {
    echo "Comparing contents of the source and destination containers..."
    
    # Sort the contents of the source and destination files before diffing
    sort "${source_storage_account_name}_${container_name}_${directory_path}_contents" > sorted_source_contents.txt
    sort "${destination_storage_account_name}_${container_name}_${directory_path}_contents" > sorted_destination_contents.txt
    
    # Perform the diff on the sorted files
    diff sorted_source_contents.txt sorted_destination_contents.txt > diff_output.txt
    
    if [ -s diff_output.txt ]; then
        echo "Differences found between source and destination containers, starting to sync..."
        echo "Differences:"
        cat diff_output.txt
        echo "Syncing the differences..."
        sync_container_contents
    else
        echo "No differences found between source and destination containers."
    fi
    
    # Clean up temporary sorted files
    rm -f sorted_source_contents.txt sorted_destination_contents.txt diff_output.txt
}

# Start syncing the container contents
echo "$(date) Starting the Sync process..."
sync_container_contents

# fetch_container_contents
# summarize_blobs
# diff_between_containers
echo "$(date) Sync process completed."