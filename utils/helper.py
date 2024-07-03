## this function shifts older values to the top by one
## and attach fetaures to end
def shift_top_inplace(memory, features):
    memory[0, 0:-1, :] = memory[0, 1:, :]
    memory[0, -1, :] = features