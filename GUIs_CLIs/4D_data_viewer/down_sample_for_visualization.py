import h5py

file_path = "/nsls2/data/hex/proposals/2025-2/pass-318635/edxd/converted/scan_00004/scan_00004.h5"
output_file = "/nsls2/data/hex/proposals/2025-2/pass-318635/edxd/converted/scan_00004/scan_00004_dsp.h5"

# # Open the file
with h5py.File(file_path, 'r') as f:
    # Access the specific dataset
    dset = f['data']
    print(dset.shape)
    # Get the chunk size
    chunk_size = dset.chunks
    if chunk_size:
        print(f"Chunk shape: {chunk_size}")
    else:
        print("Dataset is contiguous (not chunked).")


def downsample_with_mean(input_path, output_path, dataset_name):
    with h5py.File(input_path, 'r') as f_in:
        dset_in = f_in[dataset_name]
        in_shape = dset_in.shape  # (4096, 192, 151, 151)
        # Target shape (halving first two dimensions)
        out_shape = (in_shape[0] // 2, in_shape[1] // 2, in_shape[2], in_shape[3])
        chunks = (64, 64, 8, 8)

        with h5py.File(output_path, 'w') as f_out:
            # Create dataset with NO compression and specified chunks
            dset_out = f_out.create_dataset(dataset_name, shape=out_shape,
                                            dtype='float32',
                                            chunks=chunks, compression=None)
            # Define processing steps (multiples of chunk size for speed)
            # We process 64 output frames at a time in dim0
            # and 32 output frames at a time in dim1 (since 96 is a multiple of 32)
            step0 = 64
            step1 = 32
            print(f"Downsampling {in_shape} to {out_shape} via mean binning...")

            for i in range(0, out_shape[0], step0):
                for j in range(0, out_shape[1], step1):
                    # Determine boundaries for this block
                    out_i_end = min(i + step0, out_shape[0])
                    out_j_end = min(j + step1, out_shape[1])
                    # Map output boundaries back to input boundaries (factor of 2)
                    in_i_start, in_i_end = i * 2, out_i_end * 2
                    in_j_start, in_j_end = j * 2, out_j_end * 2
                    # 1. Read the block from disk
                    block = dset_in[in_i_start:in_i_end, in_j_start:in_j_end, :, :]
                    # 2. Reshape and compute mean
                    # Current block shape: (N, M, 151, 151)
                    # We reshape it to (N/2, 2, M/2, 2, 151, 151)
                    curr_n = block.shape[0]
                    curr_m = block.shape[1]
                    binned = block.reshape(curr_n // 2, 2, curr_m // 2, 2,
                                           in_shape[2], in_shape[3]).mean(axis=(1, 3))
                    # 3. Write binned block to output
                    dset_out[i:out_i_end, j:out_j_end, :, :] = binned.astype('float32')
                print(
                    f"Progress: {((i + step0) / out_shape[0] * 100):.1f}% complete")
    print("Process finished successfully.")


# Execute
downsample_with_mean(file_path, output_file, 'data')