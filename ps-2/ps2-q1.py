import numpy as np
#Question 1
#Entire implementation comes from class notebooks w/ 121 being my input
def get_bits(number):
    """For a NumPy quantity, return bit representation
    
    Parameters
    ----------
    
    number : NumPy value
        value to convert into list of bits
        
    Returns
    -------
    
    bits : list
       list of 0 and 1 values, highest to lowest significance
"""
    bytes = number.tobytes()
    bits = []
    for byte in bytes:
        bits = bits + np.flip(np.unpackbits(np.uint8(byte)), np.uint8(0)).tolist()
    return list(reversed(bits))
bitlist=str(get_bits(np.uint8(121)))
print("121 decimal -> {bitlist}".format(bitlist=bitlist))