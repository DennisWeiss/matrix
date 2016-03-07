matrix
======

A Javascript Library to perform basic matrix operations using the functional nature of Javascript

###Usage
```javascript
var a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
var A = matrix(a);
```

###Operations
####1. Identity
```javascript
A().get(); //returns [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
```
####2. Row
```javascript
A(0).get(); // returns [1, 2, 3]
```
####3. Column
```javascript
A([], 0).get(); // returns [[1][4][7]]
```
####4. Element
```javascript
A(1, 2).get(); // returns 3
```
####5. Range*
```javascript
A([1,2]).get(); // returns [[4, 5, 6], [7, 8, 9]]
A([],[1,2]).get(); // returns [[2, 3], [5, 6], [8, 9]]
A([1,2],[1,2]).get(); // returns [[5, 6], [8, 9]]
A([2,1],[]).get(); // returns [[7, 8, 9], [4, 5 ,6]]
A([],[2,1]).get(); // returns [[3, 2], [6, 5], [9, 8]]
A([2,1],[2,1]).get(); // returns [[9, 8], [6, 5]]
```
####6. Size
```javascript
A().size(); //returns [3, 3]
A([],0).size(); // returns [3, 1]
A(0).size(); // returns [3]
```
####7. Change*
```javascript
A(0).set(0); // returns [[0, 0, 0], [4, 5, 6], [7, 8, 9]]
A([], 0).set(0); // returns [[0, 2, 3], [0, 5, 6], [0, 8, 9]]
A([1,2]).set(4); // returns [[1, 2, 3], [4, 4, 4], [4, 4, 4]]
A([], [1,2]).set(1); // returns [[1, 1, 1], [4, 1, 1], [7, 1, 1]]
```
####8. Addition*
```javascript
var B = matrix([[3, 4, 5], [6, 7, 8], [9, 10, 11]]);
A().add(B); // returns [[4, 6, 8],[10, 12, 14], [16, 18, 20]]
```
####9. Subtraction*
```javascript
B().sub(A); // returns [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
```
####10. Multiplication*
```javascript
A().mul(B); // returns [[3, 8, 15], [24, 35, 48], [56, 80, 99]]
```
####11. Division*
```javascript
A().div(B); // returns [[0.33, 0.5, 0.6], [0.66, 0.71, 0.75], [0.77, 0.8, 0.81]]
```
####12. Product*
```javascript
A().prod(B); // returns [[42, 48, 54], [96, 111, 126], [150, 174, 198]]
```
####13. Transpose*
```javascript
A().trans(); // returns [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
```
####14. Determinant*
```javascript
A().det(); // returns 0
```
####15. Inverse*
Should be invertible
```javascript
M = matrix([[1, 3, 3], [1, 4, 3], [1, 3 ,4]]
M().inv(); // returns [[7, -3, -3], [-1, 1, 0], [-1, 0 ,1]]
```

__* Under Development__