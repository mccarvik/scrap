import pdb

roman = {
    1000 : "M",
    900 : "CM",
    500 : "D",
    400 : "CD",
    100 : "C",
    90 : "XC",
    50 : "L",
    40 : "XL",
    10 : "X",
    9 : "IX",
    5 : "V",
    4 : "IV",
    1 : "I"
}

def write_roman(num):
    return "".join([a for a in roman_num(num)])

def roman_num(num):
    for r in sorted(list(roman.keys()), reverse=True):
        x, y = divmod(num, r)
        yield roman[r] * x
        num -= (r * x)
        if num > 0:
            roman_num(num)
        else:
            break

print(write_roman(26))
print(write_roman(1987))

# import sys
# print (sys.argv)


# with open("tp.txt") as rad: #rad is a file object for read
# for line in rad:
# print("line: " + line.rstrip('\n'))