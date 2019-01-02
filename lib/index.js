'use strict'

const rational = require('./rational')
const merge = require('./merge')

/**
 * Pass a 2-dimensional array that will return a function accepting indices to access the matrix
 *
 * @param mat array that initializes the matrix
 * @returns function with the array initialized and access to method that perform operations on the matrix
 */
function matrix(mat) {
    if (!Array.isArray(mat)) {
        throw new Error('Input should be of type array')
    }
    let _matrix = function () {
        let args = (arguments.length === 1 ? [arguments[0]] : Array.apply(null, arguments))
        return read(mat, args)
    }
    return Object.assign(_matrix, _mat(mat))
}


/**
 * Private function that returns an object containing methods
 * that perform operations on the matrix
 *
 * @param mat array that initializes the matrix
 * @returns object of methods performing matrix operations
 */
function _mat(mat) {
    return {
        size: () => size(mat),
        add: (operand) => operate(mat, operand, addition),
        sub: (operand) => operate(mat, operand, subtraction),
        mul: (operand) => operate(mat, operand, multiplication),
        div: (operand) => operate(mat, operand, division),
        prod: (operand) => prod(mat, operand),
        vectorProd: vector => vectorProd(mat, vector),
        trans: () => trans(mat),
        item: (i, j) => mat[i][j],
        set: function () {
            let args = (arguments.length === 1 ? [arguments[0]] : Array.apply(null, arguments))
            return {
                to: (val) => replace(mat, val, args)
            }
        },
        det: () => determinant(mat),
        inv: () => invert(mat),
        merge: merge(mat),
        map: (func) => map(mat, func),
        isInRef: () => isInRef(mat),
        isInRref: () => isInRref(mat),
        ref: () => ref(mat),
        rref: () => rref(mat),
        solve: b => solve(mat, b),
        powerIter: () => powerIter(mat),
        eigenDecompositionOfSymmetricMatrix: () => eigenDecompositionOfSymmetricMatrix(mat),
        diagonal: () => diagonal(mat),
        trace: () => trace(mat),
        svd: () => svd(mat)
    }
}

module.exports = matrix


/**
 * Calculates the size of the array across each dimension
 *
 * @param mat input matrix that initialized the function
 * @returns size of the matrix as an array
 */
function size(mat) {
    let s = []
    while (Array.isArray(mat)) {
        s.push(mat.length)
        mat = mat[0]
    }
    return s
}


function diagonal(mat) {
    const diagonalArr = []
    for (let i = 0; i < Math.min(mat.length, mat[0].length); i++) {
        diagonalArr.push(mat[i][i])
    }
    return diagonalArr
}


function trace(mat) {
    return diagonal(mat).reduce((a, b) => a + b)
}


/**
 * Private function to calculate the dimensions of the matrix
 *
 * @param mat input matrix that initializes the function
 * @returns integer indicating the number of dimensions
 */
function dimensions(mat) {
    return size(mat).length
}


/**
 * Outputs the original matrix or a particular element or a matrix that is part of the original
 *
 * @param mat input matrix that initializes the function
 * @param args indices to access one or more array elements
 * @returns array or single element
 */
function read(mat, args) {
    if (args.length === 0) {
        return mat
    } else {
        return extract(mat, args)
    }
}


/**
 * Private function to extract a single element or a matrix that is part of the original
 *
 * @param mat input matrix that initializes the function
 * @param args indices to access one or more array elements
 * @returns array or single element
 */
function extract(mat, args) {
    let dim = dimensions(mat)
    for (let i = 0; i < dim; i++) {
        let d = args[i]
        if (d === undefined) {
            break
        }
        if (Array.isArray(d)) {
            // if an element of args is an array, more extraction is needed
            mat = extractRange(mat, d, i)
        } else if (Number.isInteger(d)) {
            if (dimensions(mat) > 1 && i > 0) {
                mat = mat.map(function (elem) {
                    return [elem[d]]
                })
            } else {
                mat = mat[d]
            }
        }
    }
    return mat
}


/**
 * Private function to extract a portion of the array based on the specified range
 *
 * @param mat input matrix that initialized the function
 * @param arg single argument containing the range specified as an array
 * @param ind the current index of the arguments while extracting the matrix
 * @returns array from the specified range
 */
function extractRange(mat, arg, ind) {
    if (!arg.length) {
        return mat
    } else if (arg.length === 2) {
        let reverse = arg[0] > arg[1]
        let first = (!reverse) ? arg[0] : arg[1]
        let last = (!reverse) ? arg[1] : arg[0]
        if (dimensions(mat) > 1 && ind > 0) {
            return mat.map(function (elem) {
                if (reverse) {
                    return elem.slice(first, last + 1).reverse()
                }
                return elem.slice(first, last + 1)
            })
        } else {
            mat = mat.slice(first, last + 1)
            return (reverse && mat.reverse()) || mat
        }
    }
}


/**
 * Replaces the specified index in the matrix with the specified value
 *
 * @param mat input matrix that initialized the function
 * @param value specified value that replace current value at index or indices
 * @param args index or indices passed in arguments to initialized function
 * @returns replaced matrix
 */
function replace(mat, value, args) { //TODO: Clean this function up
    let result = clone(mat)
    let prev = args[0]
    let start = prev[0] || 0
    let end = prev[1] && prev[1] + 1 || mat.length
    if (!Array.isArray(prev) && args.length === 1) {
        result[prev].fill(value)
    } else if (args.length === 1) {
        for (let ind = start; ind < end; ind++) {
            result[ind].fill(value)
        }
    }
    for (let i = 1; i < args.length; i++) {
        let first = Array.isArray(args[i]) ? args[i][0] || 0 : args[i]
        let last = Array.isArray(args[i]) ? args[i][1] && args[i][1] + 1 || mat[0].length : args[i] + 1
        if (!Array.isArray(prev)) {
            result[prev].fill(value, first, last)
        } else {
            for (let ind = start; ind < end; ind++) {
                result[ind].fill(value, first, last)
            }
        }
    }
    return result
}


/**
 * Operates on two matrices of the same size
 *
 * @param mat input matrix that initialized the function
 * @param operand second matrix with which operation is performed
 * @param operation function performing the desired operation
 * @returns result of the operation
 */
function operate(mat, operand, operation) {
    let result = []
    let op = operand()

    for (let i = 0; i < mat.length; i++) {
        let op1 = mat[i]
        let op2 = op[i]
        result.push(op1.map(function (elem, ind) {
            return operation(elem, op2[ind])
        }))
    }

    return result
}


/**
 * Finds the product of two matrices
 *
 * @param mat input matrix that initialized the function
 * @param operand second matrix with which operation is performed
 * @returns the product of the two matrices
 */
function prod(mat, operand) {
    let op1 = mat
    let op2 = operand()
    let size1 = size(op1)
    let size2 = size(op2)
    let result = []
    if (size1[1] === size2[0]) {
        for (let i = 0; i < size1[0]; i++) {
            result[i] = []
            for (let j = 0; j < size2[1]; j++) {
                for (let k = 0; k < size1[1]; k++) {
                    if (result[i][j] === undefined) {
                        result[i][j] = 0
                    }
                    result[i][j] += multiplication(op1[i][k], op2[k][j])
                }
            }
        }
    }
    return result
}


function vectorProd(mat, vector) {
    if (mat[0].length !== vector.length) {
        throw 'matrix and vector don\'t fit'
    }
    const rationalized = rationalize(mat)
    const rationalizedVector = rationalizeVec(vector)
    const width = rationalized[0].length
    const height = rationalized.length
    const product = []
    for (let i = 0; i < height; i++) {
        let sum = rational(0, 1)
        for (let j = 0; j < width; j++) {
            sum = sum.add(rationalized[i][j].mul(rationalizedVector[j]))
        }
        product.push(sum)
    }
    return derationalizeVec(product)
}


/**
 * Returns the transpose of a matrix, swaps rows with columns
 *
 * @param mat input matrix that initialized the function
 * @returns a matrix with rows and columns swapped from the original matrix
 */
function trans(mat) {
    let input = mat
    let s = size(mat)
    let output = []
    for (let i = 0; i < s[0]; i++) {
        for (let j = 0; j < s[1]; j++) {
            if (Array.isArray(output[j])) {
                output[j].push(input[i][j])
            } else {
                output[j] = [input[i][j]]
            }
        }
    }
    return output
}

/**
 * Private method to clone the matrix
 *
 * @param mat input matrix that initialized the function
 * @returns cloned matrix
 */
function clone(mat) {
    let result = []
    for (let i = 0; i < mat.length; i++) {
        result.push(mat[i].slice(0))
    }
    return result
}

/**
 * Performs addition
 *
 * @param op1 first operand
 * @param op2 second operand
 * @returns result
 */
function addition(op1, op2) {
    return op1 + op2
}

/**
 * Performs subtraction
 *
 * @param op1 first operand
 * @param op2 second operand
 * @returns result
 */
function subtraction(op1, op2) {
    return op1 - op2
}

/**
 * Performs multiplication
 *
 * @param op1 first operand
 * @param op2 second operand
 * @returns result
 */
function multiplication(op1, op2) {
    return op1 * op2
}

/**
 * Performs division
 *
 * @param op1 first operand
 * @param op2 second operand
 * @returns result
 */
function division(op1, op2) {
    return op1 / op2
}


/**
 * Computes the determinant using row reduced echelon form
 * Works best if the elements are integers or rational numbers
 * The matrix must be a square
 *
 * @param mat input matrix that initialized the function
 * @returns determinant value as a number
 */
function determinant(mat) {
    let rationalized = rationalize(mat)
    let siz = size(mat)
    let det = rational(1)
    let sign = 1

    for (let i = 0; i < siz[0] - 1; i++) {
        for (let j = i + 1; j < siz[0]; j++) {
            if (rationalized[j][i].num === 0) {
                continue
            }
            if (rationalized[i][i].num === 0) {
                interchange(rationalized, i, j)
                sign = -sign
                continue
            }
            let temp = rationalized[j][i].div(rationalized[i][i])
            temp = rational(Math.abs(temp.num), temp.den)
            if (Math.sign(rationalized[j][i].num) === Math.sign(rationalized[i][i].num)) {
                temp = rational(-temp.num, temp.den)
            }
            for (let k = 0; k < siz[1]; k++) {
                rationalized[j][k] = temp.mul(rationalized[i][k]).add(rationalized[j][k])
            }
        }
    }

    det = rationalized.reduce((prev, curr, index) => prev.mul(curr[index]), rational(1))

    return sign * det.num / det.den
}

/**
 * Checks whether a given matrix is in row echelon form
 *
 * @param mat input matrix
 * @returns {boolean} whether matrix is in row echelon form
 */
function isInRef(mat) {
    let leadingEntry = -1
    for (let i = 0; i < mat.length; i++) {
        let isZeroRow = true
        for (let j = 0; j < mat[i].length; j++) {
            if (mat[i][j] !== 0) {
                if (j > leadingEntry) {
                    isZeroRow = false
                    leadingEntry = j
                    break
                } else {
                    return false
                }
            }
        }
        if (isZeroRow) {
            leadingEntry = mat[i].length
        }
    }
    return true
}

/**
 * Returns the matrix in row echelon form
 * @param mat input matrix
 * @returns {matrix} in row echelon form
 */
function ref(mat) {
    const rationalized = rationalize(mat)
    refHelper(rationalized)
    return derationalize(rationalized)
}

function refHelper(rationalized, pivot = {i: 0, j: 0}) {
    if (!(pivot.i === rationalized.length - 1 && pivot.j === rationalized[0].length - 1 || pivot.j >= rationalized[0].length)) {
        const width = rationalized[0].length
        const height = rationalized.length

        let isZeroCol = true
        for (let i = pivot.i; i < height; i++) {
            if (rationalized[i][pivot.j].num !== 0) {
                isZeroCol = false
                if (i > pivot.i) {
                    interchange(rationalized, pivot.i, i)
                }
                break
            }
        }

        if (!isZeroCol) {
            for (let i = pivot.i + 1; i < height; i++) {
                const factor = rationalized[i][pivot.j].div(rationalized[pivot.i][pivot.j])
                for (let j = pivot.j; j < width; j++) {
                    rationalized[i][j] = rationalized[i][j].sub(factor.mul(rationalized[pivot.i][j]))
                }
            }
            refHelper(rationalized, {i: pivot.i + 1, j: pivot.j + 1})
        } else {
            refHelper(rationalized, {i: pivot.i, j: pivot.j + 1})
        }
    }
}

/**
 * Return true if the matrix is in reduced row echelon form
 * @param mat input matrix
 * @returns {boolean} whether the matrix is in reduced row echelon form
 */
function isInRref(mat) {
    const rationalized = rationalize(mat)
    const width = rationalized[0].length
    const height = rationalized.length

    for (let i = 0; i < height; i++) {
        for (let j = 0; j < width; j++) {
            if (rationalized[i][j].num !== 0) {
                if (rationalized[i][j].num === rationalized[i][j].den) {
                    for (let k = i; k >= 0; k--) {
                        if (rationalized[k][j].num !== 0) {
                            return false
                        }
                    }
                } else {
                    return false
                }
            }
        }
    }
    return true
}

function rref(mat) {
    const rationalized = rationalize(mat)
    refHelper(rationalized)
    refToRref(rationalized)
    return derationalize(rationalized)
}

function refToRref(rationalized) {
    normalizeLeadingEntries(rationalized)
    ensureZerosInPivotColumns(rationalized)
}

function normalizeLeadingEntries(rationalized) {
    const width = rationalized[0].length
    const height = rationalized.length
    for (let i = height - 1; i >= 0; i--) {
        let leadingEntry
        for (let j = 0; j < width; j++) {
            if (leadingEntry) {
                rationalized[i][j] = rationalized[i][j].div(leadingEntry)
            } else if (rationalized[i][j].num !== 0) {
                leadingEntry = rationalized[i][j]
                rationalized[i][j] = rationalized[i][j].div(rationalized[i][j])
            }
        }
    }
}

function ensureZerosInPivotColumns(rationalized) {
    const width = rationalized[0].length
    const height = rationalized.length
    for (let i = height - 1; i >= 0; i--) {
        let pivotColumn
        for (let j = 0; j < width; j++) {
            if (rationalized[i][j].num !== 0) {
                pivotColumn = j
                break
            }
        }
        if (pivotColumn != null) {
            for (let j = i - 1; j >= 0; j--) {
                const factor = rationalized[j][pivotColumn]
                for (let k = 0; k < width; k++) {
                    rationalized[j][k] = rationalized[j][k].sub(factor.mul(rationalized[i][k]))
                }
            }
        }
    }
}

function deepCopy(mat) {
    const copy = []
    for (let i = 0; i < mat.length; i++) {
        const row = []
        for (let j = 0; j < mat[i].length; j++) {
            row.push(mat[i][j])
        }
        copy.push(row)
    }
    return copy
}

function constructAugmentedMatrix(A, b) {
    if (A.length !== b.length) {
        throw 'Dimensions of matrix and vector do not match!'
    }
    const copy = deepCopy(A)
    for (let i = 0; i < b.length; i++) {
        copy[i].push(b[i])
    }
    return copy
}

function subMatrixIsIdentity(mat, n) {
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            if (i === j && mat[i][j] !== 1 || i !== j && mat[i][j] !== 0) {
                return false
            }
        }
    }
    return true
}

function getColumnOfMatrix(mat, column) {
    if (column >= mat[0].length) {
        throw 'Column index is out of bounds!'
    }
    const col = []
    for (let i = 0; i < mat.length; i++) {
        col.push(mat[i][column])
    }
    return col
}

/**
 * Solves a linear system of equations. Parameters are provided as written in matrix vector notation: Ax=b
 * @param A matrix
 * @param b constant vector
 * @returns vector solution vector
 */
function solve(A, b) {
    const m = A.length
    const n = A[0].length
    const k = b.length
    if (k !== m) {
        throw 'Dimensions of matrix and vector do not match!'
    }
    if (m !== n) {
        throw 'Matrix is not a square matrix!'
    }
    const augmentedInRref = matrix(constructAugmentedMatrix(A, b)).rref()
    if (subMatrixIsIdentity(augmentedInRref, n)) {
        return getColumnOfMatrix(augmentedInRref, n)
    }
    return null
}

/**
 * Interchanges one row index with another on passed matrix
 *
 * @param mat input matrix
 * @param ind1 one of the row indices to exchange
 * @param ind2 one of the row indices to exchange
 */
function interchange(mat, ind1, ind2) {
    let temp = mat[ind1]
    mat[ind1] = mat[ind2]
    mat[ind2] = temp
}

/**
 * Inverts the input square matrix using row reduction technique
 * Works best if the elements are integers or rational numbers
 * The matrix has to be a square and non-singular
 *
 * @param mat input matrix
 * @returns inverse of the input matrix
 */
function invert(mat) {
    let rationalized = rationalize(mat)
    let siz = size(mat)
    let result = rationalize(identity(siz[0]))

    // Row Reduction
    for (let i = 0; i < siz[0] - 1; i++) {
        if (rationalized[i][i].num === 0) {
            interchange(rationalized, i, i + 1)
            interchange(result, i, i + 1)
        }
        if (rationalized[i][i].num !== 1 || rationalized[i][i] !== 1) {
            let factor = rational(rationalized[i][i].num, rationalized[i][i].den)
            for (let col = 0; col < siz[1]; col++) {
                rationalized[i][col] = rationalized[i][col].div(factor)
                result[i][col] = result[i][col].div(factor)
            }
        }
        for (let j = i + 1; j < siz[0]; j++) {
            if (rationalized[j][i].num === 0) {
                // skip as no row elimination is needed
                continue
            }

            let temp = rational(-rationalized[j][i].num, rationalized[j][i].den)
            for (let k = 0; k < siz[1]; k++) {
                rationalized[j][k] = temp.mul(rationalized[i][k]).add(rationalized[j][k])
                result[j][k] = temp.mul(result[i][k]).add(result[j][k])
            }
        }
    }

    // Further reduction to convert rationalized to identity
    let last = siz[0] - 1
    if (rationalized[last][last].num !== 1 || rationalized[last][last] !== 1) {
        let factor = rational(rationalized[last][last].num, rationalized[last][last].den)
        for (let col = 0; col < siz[1]; col++) {
            rationalized[last][col] = rationalized[last][col].div(factor)
            result[last][col] = result[last][col].div(factor)
        }
    }

    for (let i = siz[0] - 1; i > 0; i--) {
        for (let j = i - 1; j >= 0; j--) {
            let temp = rational(-rationalized[j][i].num, rationalized[j][i].den)
            for (let k = 0; k < siz[1]; k++) {
                rationalized[j][k] = temp.mul(rationalized[i][k]).add(rationalized[j][k])
                result[j][k] = temp.mul(result[i][k]).add(result[j][k])
            }
        }
    }

    return derationalize(result)
}

/**
 * Applies a given function over the matrix, elementwise. Similar to Array.map()
 * The supplied function is provided 4 arguments:
 * the current element,
 * the row index,
 * the col index,
 * the matrix.
 *
 * @param mat input matrix
 * @returns matrix of same dimensions with values altered by the function.
 */
function map(mat, func) {
    const s = size(mat)
    const result = []
    for (let i = 0; i < s[0]; i++) {
        if (Array.isArray(mat[i])) {
            result[i] = []
            for (let j = 0; j < s[1]; j++) {
                result[i][j] = func(mat[i][j], i, j, mat)
            }
        } else {
            result[i] = func(mat[i], i, 0, mat)
        }
    }
    return result
}

/**
 * Converts a matrix of numbers to all rational type objects
 *
 * @param mat input matrix
 * @returns matrix with elements of type rational
 */
function rationalize(mat) {
    return mat.map(row => row.map((ele) => rational(ele)))
}

/**
 * Converts a vector of numbers to all rational type objects
 *
 * @param vector input vector
 * @returns vector with elements of type rational
 */
function rationalizeVec(vector) {
    return vector.map(elm => rational(elm))
}

/**
 * Converts a rationalized matrix to all numerical values
 *
 * @param mat input matrix
 * @returns matrix with numerical values
 */
function derationalize(mat) {
    let derationalized = []
    mat.forEach((row, ind) => {
        derationalized.push(row.map((ele) => ele.num / ele.den))
    })
    return derationalized
}

/**
 * Converts a rationalized vector to all numerical values
 * @param vector input vector
 * @returns vector with numerical values
 */
function derationalizeVec(vector) {
    return vector.map(elm => elm.num / elm.den)
}

/**
 * Generates a square matrix of specified size all elements with same specified value
 *
 * @param size specified size
 * @param val specified value
 * @returns square matrix
 */
function generate(size, val) {
    let dim = 2
    while (dim > 0) {
        var arr = []
        for (var i = 0; i < size; i++) {
            if (Array.isArray(val)) {
                arr.push(Object.assign([], val))
            } else {
                arr.push(val)
            }
        }
        val = arr
        dim -= 1
    }
    return val
}

/**
 * Generates an identity matrix of the specified size
 *
 * @param size specified size
 * @returns identity matrix
 */
function identity(size) {
    let result = generate(size, 0)
    result.forEach((row, index) => {
        row[index] = 1
    })
    return result
}

function abs(vector) {
    return Math.sqrt(vector.reduce((a, b) => a + b * b, 0))
}

function normalized(vector) {
    const absValue = abs(vector)
    return vector.map(x => x / absValue)
}

function eigenValueFromEigenVector(mat, eigenVector) {
    const prod = vectorProd(mat, eigenVector)
    let sum = 0
    for (let i = 0; i < eigenVector.length; i++) {
        sum += prod[i] / eigenVector[i]
    }
    return sum / eigenVector.length
}

/**
 * Returns the eigenvector with the largest eigenvalue
 *
 * @param mat input matrix
 * @param {number} [iterations] Number of iterations. Default is 10000.
 * @returns eigenpair with the greatest eigenvalue
 */
function powerIter(mat, iterations=10000) {
    const width = mat[0].length
    let b = [...Array(width)].map(_ => Math.random())
    for (let i = 0; i < iterations; i++) {
        b = normalized(vectorProd(mat, b))
    }
    return {
        eigenVector: b,
        eigenValue: eigenValueFromEigenVector(mat, b)
    }
}

function sumOfNonDiagonalEntries(mat) {
    if (mat.length !== mat[0].length) {
        throw "Matrix is not a square matrix"
    }
    const n = mat.length
    let sum = 0
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            if (i !== j) {
                sum += mat[i][j]
            }
        }
    }
    return sum
}

function getRotationAngle(mat, i, j) {
    if (mat[i][i] === mat[j][j]) {
        if (mat[i][j] > 0) {
            return Math.PI / 4
        }
        return -Math.PI / 4
    }
    return 0.5 * Math.atan(2 * mat[i][j] / (mat[j][j] - mat[i][i]))
}

function getGivensRotationMatrix(mat, i, j) {
    const height = mat.length
    const width = mat[0].length
    const theta = getRotationAngle(mat, i, j)
    const rotationMatrix = []
    for (let k = 0; k < height; k++) {
        const rotationRow = []
        for (let l = 0; l < width; l++) {
            let elm = 0
            if (k === l) {
                if (k === i || k === j) {
                    elm = Math.cos(theta)
                } else {
                    elm = 1
                }
            } else if (k === i && j === l && i > j || k === j && i === l && j > i) {
                elm = Math.sin(theta)
            } else if (k === j && i === l && j < i || k === i && j === l && i < j) {
                elm = -Math.sin(theta)
            }
            rotationRow.push(elm)
        }
        rotationMatrix.push(rotationRow)
    }
    return rotationMatrix
}

function do_givens_rotation(mat, v) {
    const n = mat.length
    let largest = [1, 0, Math.abs(mat[1][0])]
    for (let i = 1; i < n; i++) {
        for (let j = 0; j < i; j++) {
            if (Math.abs(mat[i][j]) > largest[2]) {
                largest = [i, j, Math.abs(mat[i][j])]
            }
        }
    }
    const rotationMatrix = getGivensRotationMatrix(mat, largest[0], largest[1])
    const b = matrix(matrix(trans(rotationMatrix)).prod(matrix(mat))).prod(matrix(rotationMatrix))
    const v2 = matrix(v).prod(matrix(rotationMatrix))
    return [b, v2]
}

function eigenDecompositionOfSymmetricMatrix(mat) {
    const mat2 = matrix(mat)
    console.log(mat2.size())
    if (mat2.size()[0] !== mat2.size()[1]) {
        throw "Matrix is not a square matrix"
    }
    // TODO: add symmetric check
    const n = mat2.size()[0]
    let b = mat
    let v = identity(n)
    for (let i = 0; i < 50; i++) {
        [b, v] = do_givens_rotation(b, v)
    }
    return [v, diagonal(b), trans(v)]
}

function svd(a) {
    const height = a.length
    const width = a[0].length
    const at = trans(a)
    const ata = matrix(at).prod(matrix(a))
    console.log('ata', ata)
    const aat = matrix(a).prod(matrix(at))
    const ataEigenDecomposition = eigenDecompositionOfSymmetricMatrix(ata)
    const aatEigenDecomposition = eigenDecompositionOfSymmetricMatrix(aat)
    return [aatEigenDecomposition[0], height > width ?
        ataEigenDecomposition[1] :
        aatEigenDecomposition[1].map(eigenvalue => eigenvalue > 0 ? Math.sqrt(eigenvalue) : 0), ataEigenDecomposition[2]]
}

