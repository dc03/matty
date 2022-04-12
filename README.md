# matty

Matty is a small C++ library consisting of two classes:
- `matty::DynamicMatrix`: a dynamically-sized matrix of either integers or floats
- `matty:FixedMatrix4`: a 4x4 matrix of either integers or floats

## Example

### Code

```cpp
DynamicMatrix m3{{2, 7}, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
std::cout << "\nm3:--------\n";
print(m3);

DynamicMatrix m4{{7, 3}, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
std::cout << "\nm4:--------\n";
print(m4);

std::cout << "\nm5:\n--------\n";
DynamicMatrix m5 = m3.mul(m4);
print(m5);
```

### Output

```
m3:--------
1 2 3 4 5 6 7 
8 9 10 11 12 13 14 

m4:--------
5 6 7 
8 9 10 
11 12 13 
14 15 16 
17 18 19 
20 21 22 
23 24 25 

m5:
--------
476 504 532 
1162 1239 1316 
```

### Code

```cpp
FixedMatrix4<float> x6 = FixedMatrix4<float>{
    1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f, 9.5f, 10.5f, 11.5f, 12.5f, 13.5f, 14.5f, 15.5f, 16.5f};
FixedMatrix4<float> x7 = FixedMatrix4<float>::identity().mul_self(x6).add_self(1.0f).mul_self(3.0f);
FixedMatrix4<float> x8 = x6.mul(x7);

std::cout << "\nx6:--------\n";
print(x6);
std::cout << "\nx7:--------\n";
print(x7);
std::cout << "\nx8:--------\n";
print(x8);
```

### Output

```
x6:--------
1.5 2.5 3.5 4.5 
5.5 6.5 7.5 8.5 
9.5 10.5 11.5 12.5 
13.5 14.5 15.5 16.5 

x7:--------
43.5 46.5 49.5 52.5 
31.5 34.5 37.5 40.5 
19.5 22.5 25.5 28.5 
7.5 10.5 13.5 16.5 

x8:--------
366 402 438 474 
774 858 942 1026 
1182 1314 1446 1578 
1590 1770 1950 2130 
```

### Code:

```cpp
    DynamicMatrix<int> m10 = DynamicMatrix<int>::identity(4, 7);
    std::cout << "\nm10:--------\n";
    print(m10);
    std::cout << "\nm10 + 2 * 5 - 1:--------\n";
    print((m10 + 2) * 5 - 1);
```

### Output:

```
m10:--------
1 0 0 0 0 0 0 
0 1 0 0 0 0 0 
0 0 1 0 0 0 0 
0 0 0 1 0 0 0 

m10 + 2 * 5 - 1:--------
14 9 9 9 9 9 9 
9 14 9 9 9 9 9 
9 9 14 9 9 9 9 
9 9 9 14 9 9 9 
```

## License

Matty is licensed under the MIT license.