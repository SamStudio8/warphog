__global__ void hamming_distance(char* msa, int num_msa, int l, unsigned short* d, int p, int n, unsigned short* idx_map, unsigned short* idy_map) {

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    // Push threadId to account for PAIRS_PER_THREAD (p)
    //threadId += (blockId * (blockDim.x*blockDim.y) * (p-1)) + ((threadIdx.x + (threadIdx.y * blockDim.x)) * (p-1));

    int count = 0;
    while(count < p){

        int idx = idx_map[threadId];
        int idy = idy_map[threadId];

        if(threadId < n){
            //printf("HELLO I AM THE GPU: thread %d, idx %d, idy %d\\n", threadId, idx, idy);

            unsigned int idd = threadId;

            int base_a = 0;
            int base_b = 0;

            if(idx == idy){
                d[idd] = 0;
            }
            else{

                int msa_a = l*idx;
                int msa_b = l*idy;

                //printf("HELLO I AM THE GPU: block %d, thread %d, p %d, idx %d, idy %d, idd %d\\n", blockId, threadIdx.x, count, idx, idy, idd);

                int i = 0;
                int distance = 0;
                while(i < l){
                    base_a = ord_lookup[msa[msa_a + i]];
                    base_b = ord_lookup[msa[msa_b + i]];

                    // No need for if statement, just add the edit distances directly
                    distance += equivalent_lookup[base_a][base_b];

                    //printf("HELLO I AM ON THE GPU: seq_a@%d=%d(%c) seq_b@%d=%d(%c), lookup[%d][%d]=%d: diffs=%d \\n", msa_a+i, msa[msa_a+i], alphabet[msa[msa_a+i]], msa_b+i, msa[msa_b+i], alphabet[msa[msa_b+i]], msa[msa_a+i], msa[msa_b+i], equivalent_lookup[msa[msa_a+i]][msa[msa_b+i]], *d);
                    i++;
                }
                d[idd] = distance;
                //d[idd] += 1;
            }
        }
        count++;
        threadId += 1;
    }
}
