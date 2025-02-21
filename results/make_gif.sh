#!/bin/bash

# Set the input directory
INPUT_DIR="/results/dataset"

# Create a temporary directory for the image files
mkdir -p temp_frames

# Convert each PDF to PNG
# Adjust the range to match the actual number of files
for i in $(seq 0 44); do
    # Add leading zeros to ensure proper sorting
    number=$(printf "%03d" $i)
    # Adjust the input file name to match the actual file names
    input_file="${INPUT_DIR}/Average_Layer_${i}_dataset_3.pdf"
    
    # Check if file exists
    if [ ! -f "$input_file" ]; then
        echo "Warning: File not found: $input_file"
        continue
    fi
    
    # Convert PDF to PNG with high quality
    pdftoppm -png -r 300 "$input_file" "temp_frames/frame_${number}"
    
    # Rename the output (pdftoppm adds -1 to the filename)
    mv "temp_frames/frame_${number}-1.png" "temp_frames/frame_${number}.png"
done

# Create GIF using ffmpeg
ffmpeg -framerate 2 -pattern_type glob -i 'temp_frames/frame_*.png' \
    -vf "scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
    -loop 0 output.gif

# Clean up temporary files
rm -r temp_frames

echo "GIF creation complete! Output saved as output.gif"
