# for i in range(50,1,-1):
#     print(i)

my_list = [10, 20, 30, 40, 50, 60, 70]
odd_index_elements = [my_list[i] for i in range(1, len(my_list), 2)]
print(odd_index_elements)