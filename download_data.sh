#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -k api_key -p path"
   echo -e "\t-api_key API key for CEDA."
   echo -e "\t-p Path to download the data to."
   exit 1 # Exit script after printing help
}

while getopts "k:p:" opt
do
   case "$opt" in
      k ) api_key="$OPTARG" ;;
      p ) output_path="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction if API key is not provided
if [ -z "$api_key" ] 
then
   echo "API key not provided";
   helpFunction
fi

# Set the output path to the default value if it is not provided
if [ -z "$output_path" ]
then
   output_path="data"
fi

# Download the data
echo "Downloading data to $output_path. This may take a while..."
sleep 2

wget -e robots=off --mirror --no-parent -r https://dap.ceda.ac.uk/badc/ukmo-midas-open/data/uk-mean-wind-obs/dataset-version-202407/ --header "Authorization: Bearer $api_key" -P $output_path

echo "Download complete"