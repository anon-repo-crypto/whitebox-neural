# L_{i+1}:=((L_{i}>>alpha)+(mod) R_{i})+k, R_{i+1}:=(R_{i} << beta)+L_{i+1}
import numpy as np
from os import urandom


def WORD_SIZE():
    return (16);


def ALPHA():
    return (7);


def BETA():
    return (2);


# 二进制全为1
MASK_VAL = 2 ** WORD_SIZE() - 1;


def shuffle_together(l):
    state = np.random.get_state();
    for x in l:
        np.random.set_state(state);
        np.random.shuffle(x);


# 左移
def rol(x, k):
    return (((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)));


# 右移
def ror(x, k):
    return ((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL));


def dec_one_round(c, k):
    c0, c1 = c[0], c[1];
    # print("c[0] shape",c[0].shape)
    # print("c[1] shape",c[1].shape)
    c1 = c1 ^ c0;
    c1 = ror(c1, BETA());
    c0 = c0 ^ k;
    c0 = (c0 - c1) & MASK_VAL;
    c0 = rol(c0, ALPHA());
    # print(c1.shape)
    return (c0, c1);


def expand_key(k, t):
    ks = [0 for i in range(t)];
    ks[0] = k[len(k) - 1];
    l = list(reversed(k[:len(k) - 1]));
    for i in range(t - 1):
        l[i % 3], ks[i + 1] = enc_one_round((l[i % 3], ks[i]), i);
    return (ks);


def enc_one_round(p, k):
    c0, c1 = p[0], p[1];
    # print(c0.shape)
    c0 = ror(c0, ALPHA());
    # & MASK_VAL 模加操作
    c0 = (c0 + c1) & MASK_VAL;
    c0 = c0 ^ k;
    c1 = rol(c1, BETA());
    c1 = c1 ^ c0;
    return (c0, c1);


# Original encryption algorithm
def encrypt(p, ks):
    x, y = p[0], p[1];
    # print(x.shape)
    for k in ks:
        x, y = enc_one_round((x, y), k);
    return (x, y);


def extract_2bit_window(value, bit_positions):
    """
    Extracts a 2-bit window from a 16-bit value based on the specified bit positions.

    Args:
    - value (int): The 16-bit input value to extract from.
    - bit_positions (list): A list of 2 consecutive bit positions (e.g., [11, 10]).

    Returns:
    - int: The extracted 2-bit value, aligned to the least significant bits (LSBs).
    """
    # Ensure bit_positions is sorted in descending order (MSB to LSB)
    bit_positions = sorted(bit_positions, reverse=True)

    # Compute the mask by setting bits in the specified positions
    mask = sum(1 << bit for bit in bit_positions)

    # Align the extracted bits to the least significant bits (LSBs)
    shift_amount = min(bit_positions)
    extracted = (value & mask) >> shift_amount

    return extracted


def extract_4bit_window(value, bit_positions):
    """
    Extracts a 4-bit window from a 16-bit value based on the specified bit positions.

    Args:
    - value (int): The 16-bit input value to extract from.
    - bit_positions (list): A list of 4 consecutive bit positions (e.g., [6, 5, 4, 3]).

    Returns:
    - int: The extracted 4-bit value, aligned to the least significant bits (LSBs).
    """
    # Ensure bit_positions is sorted in descending order (MSB to LSB)
    bit_positions = sorted(bit_positions, reverse=True)

    # Compute the mask by setting bits in the specified positions
    mask = sum(1 << bit for bit in bit_positions)

    # Align the extracted bits to the least significant bits (LSBs)
    shift_amount = min(bit_positions)
    extracted = (value & mask) >> shift_amount

    return extracted


# Mask important nibbles (based on SHAP) - Window size of 2 - 8-bit output
def encrypt_and_mask_2(p, ks):
    x, y = p[0], p[1];
    # print(x.shape)
    for k in ks:
        x, y = enc_one_round((x, y), k);

    # Specify the 2-bit windows
    window1 = [13, 12]  # Set 2-bit window
    window2 = [6, 5]  # Set 2-bit window

    # Extract and combine for x
    x_part1 = extract_2bit_window(x, window1)
    x_part2 = extract_2bit_window(x, window2)
    x_extracted = (x_part1 << 2) | x_part2  # Combine the extracted parts

    # Extract and combine for y
    y_part1 = extract_2bit_window(y, window1)
    y_part2 = extract_2bit_window(y, window2)
    y_extracted = (y_part1 << 2) | y_part2  # Combine the extracted parts

    return (x_extracted, y_extracted);


# Mask important nibbles (based on SHAP) - Window size of 4 - 16-bit output
def encrypt_and_mask_4(p, ks):
    x, y = p[0], p[1];
    # print(x.shape)
    for k in ks:
        x, y = enc_one_round((x, y), k);

    # Specify the bit windows
    # window1 = [13, 12, 11, 10]
    # window2 = [6, 5, 4, 3]
    # Bad window
    window1 = [9, 8, 7, 6]
    window2 = [3, 2, 1, 0]

    # Extract and combine for x
    x_part1 = extract_4bit_window(x, window1)
    x_part2 = extract_4bit_window(x, window2)
    x_extracted = (x_part1 << 4) | x_part2  # Combine the extracted parts

    # Extract and combine for y
    y_part1 = extract_4bit_window(y, window1)
    y_part2 = extract_4bit_window(y, window2)
    y_extracted = (y_part1 << 4) | y_part2  # Combine the extracted parts

    return (x_extracted, y_extracted);


# Mask bad nibbles (based on SHAP) - 8 bit output
# def encrypt_and_mask_bad(p, ks):
#     x, y = p[0], p[1];
#     # print(x.shape)
#     for k in ks:
#         x, y = enc_one_round((x, y), k);
#
#     # Define masks for bit extraction
#     mask = 0b1111000000  # Mask for bits 9,8,7,6
#
#     # Extract and align bits
#     x_part = (x & mask) >> 6;  # We shift the 4-bit window to the 5th to 8th bit (positions 4 to 7)
#     y_part = (y & mask) >> 6;  # We shift the 4-bit window to the 5th to 8th bit (positions 4 to 7)
#
#     x_extracted = x_part
#     y_extracted = y_part
#
#     return (x_extracted, y_extracted);

def decrypt(c, ks):
    x, y = c[0], c[1];
    for k in reversed(ks):
        x, y = dec_one_round((x, y), k);
    return (x, y);


def check_testvector():
    key = (0x1918, 0x1110, 0x0908, 0x0100)
    pt = (0x6574, 0x694c)
    ks = expand_key(key, 22)
    ct = encrypt(pt, ks)
    if (ct == (0xa868, 0x42f2)):
        print("Testvector verified.")
        return (True);
    else:
        print("Testvector not verified.")
        return (False);


# convert_to_binary takes as input an array of ciphertext pairs
# where the first row of the array contains the lefthand side of the ciphertexts,
# the second row contains the righthand side of the ciphertexts,
# the third row contains the lefthand side of the second ciphertexts,
# and so on
# it returns an array of bit vectors containing the same data
# def convert_to_binary(arr):
#     X = np.zeros((4 * WORD_SIZE(),len(arr[0])),dtype=np.uint8);
#     for i in range(4 * WORD_SIZE()):
#         index = i // WORD_SIZE();
#         offset = WORD_SIZE() - (i % WORD_SIZE()) - 1;
#         X[i] = (arr[index] >> offset) & 1;
#     X = X.transpose();
#     return(X);

# def convert_to_binary(arr):
#     # print(arr.shape)
#     X = np.empty((6 * WORD_SIZE(),len(arr[0])),dtype=np.bool);
#     # print(arr[0])
#     for i in range(6 * WORD_SIZE()):
#         index = i // WORD_SIZE();
#         offset = WORD_SIZE() - (i % WORD_SIZE()) - 1;
#         X[i] = (arr[index] >> offset) & 1;
#     X = X.transpose();
#     return(X);
def convert_to_binary(l):
    n = len(l)
    k = WORD_SIZE() * n
    X = np.zeros((k, len(l[0])), dtype=np.uint8)
    for i in range(k):
        index = i // WORD_SIZE()
        offset = WORD_SIZE() - 1 - i % WORD_SIZE()
        X[i] = (l[index] >> offset) & 1
    X = X.transpose()
    return (X)


def make_train_data(n, nr, pairs=2, diff=(0x0040, 0)):
    Y = np.frombuffer(urandom(n), dtype=np.uint8);
    Y = Y & 1;
    # ratio_0 = 0.3  # Proportion of 0s
    # ratio_1 = 0.7  # Proportion of 1s
    # Generate random 0s and 1s with custom ratio
    # Y = np.random.choice([0, 1], size=n, p=[ratio_0, ratio_1])
    Y1 = np.tile(Y, pairs);
    keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1);
    keys = np.tile(keys, pairs);
    plain0l = np.frombuffer(urandom(2 * n * pairs), dtype=np.uint16);
    plain0r = np.frombuffer(urandom(2 * n * pairs), dtype=np.uint16);

    plain1l = plain0l ^ diff[0];
    plain1r = plain0r ^ diff[1];
    num_rand_samples = np.sum(Y1 == 0);

    plain1l[Y1 == 0] = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16);
    plain1r[Y1 == 0] = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16);
    ks = expand_key(keys, nr);
    # ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
    # ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);

    # Mask important nibbles data
    ctdata0l_n, ctdata0r_n = encrypt_and_mask_4((plain0l, plain0r), ks);
    ctdata1l_n, ctdata1r_n = encrypt_and_mask_4((plain1l, plain1r), ks);
    # c1 = convert_to_binary([ctdata0l_n]);
    # c2 = convert_to_binary([ctdata0r_n]);
    # c3 = convert_to_binary([ctdata1l_n]);
    # c4 = convert_to_binary([ctdata1r_n]);

    #
    # R0 = ror(ctdata0l^ctdata0r,BETA())
    # R1 = ror(ctdata1l^ctdata1r,BETA())
    # X = convert_to_binary([R0,R1,ctdata0l, ctdata0r, ctdata1l, ctdata1r]);
    # original SPECK
    # X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r]);
    # X = X.reshape(pairs,n,16*4).transpose((1,0,2))
    # Masked SPECK window 4
    ctdata0_n = (ctdata0l_n << 8) | ctdata0r_n
    ctdata1_n = (ctdata1l_n << 8) | ctdata1r_n
    X = convert_to_binary([ctdata0_n, ctdata1_n]);
    X = X.reshape(pairs, n, 16 * 2).transpose((1, 0, 2))
    # Masked SPECK window 2

    # ctdata0_n = (ctdata0l_n << 4) | ctdata0r_n
    # ctdata1_n = (ctdata1l_n << 4) | ctdata1r_n
    # # c5 = convert_to_binary([ctdata0_n]);
    # # c6 = convert_to_binary([ctdata1_n]);
    # ctdata_n = (ctdata0_n << 8) | ctdata1_n
    # X = convert_to_binary([ctdata_n]);
    # X = X.reshape(pairs,n,16).transpose((1,0,2))
    X = X.reshape(n, 1, -1)
    X = np.squeeze(X)
    return (X, Y);


# real differences data generator
def real_differences_data(n, nr, pairs=2, diff=(0x0040, 0)):
    # generate labels
    Y = np.frombuffer(urandom(n), dtype=np.uint8);
    Y = Y & 1;
    Y1 = np.tile(Y, pairs);
    # generate keys
    keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1);
    keys = np.tile(keys, pairs);
    # generate plaintexts
    plain0l = np.frombuffer(urandom(2 * n * pairs), dtype=np.uint16);
    plain0r = np.frombuffer(urandom(2 * n * pairs), dtype=np.uint16);
    # apply input difference
    plain1l = plain0l ^ diff[0];
    plain1r = plain0r ^ diff[1];
    num_rand_samples = np.sum(Y1 == 0);
    # expand keys and encrypt
    ks = expand_key(keys, nr);
    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);
    # generate blinding values
    # 加入噪声
    k0 = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16);
    k1 = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16);
    # apply blinding to the samples labelled as random
    ctdata0l[Y1 == 0] = ctdata0l[Y1 == 0] ^ k0;
    ctdata0r[Y1 == 0] = ctdata0r[Y1 == 0] ^ k1;
    ctdata1l[Y1 == 0] = ctdata1l[Y1 == 0] ^ k0;
    ctdata1r[Y1 == 0] = ctdata1r[Y1 == 0] ^ k1;
    # convert to input data for neural networks
    # R0 = ror(ctdata0l^ctdata0r,BETA())
    # R1 = ror(ctdata1l^ctdata1r,BETA())
    # X = convert_to_binary([R0,R1,ctdata0l, ctdata0r, ctdata1l, ctdata1r]);
    # X = X.reshape(pairs,n,16*6).transpose((1,0,2))
    # X = X.reshape(n,1,-1)
    # X = np.squeeze(X)

    ctdata0l = ctdata0l.reshape(pairs, n).transpose().flatten()
    ctdata0r = ctdata0r.reshape(pairs, n).transpose().flatten()
    ctdata1l = ctdata1l.reshape(pairs, n).transpose().flatten()
    ctdata1r = ctdata1r.reshape(pairs, n).transpose().flatten()
    R0 = ror(ctdata0l ^ ctdata0r, BETA())
    R1 = ror(ctdata1l ^ ctdata1r, BETA())

    X = convert_to_binary([R0, R1, ctdata0l, ctdata0r, ctdata1l, ctdata1r]);
    X = X.reshape(n, pairs * 16 * 6)

    return (X, Y);


if __name__ == "__main__":
    num_rounds = 5
    X, Y = make_train_data(4, num_rounds);