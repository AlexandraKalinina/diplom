import org.apache.commons.math.linear.*;

import java.util.Arrays;
public class MnkLiner {

    public static final int n = 2;
    public static final int N = 10;
    public static final double[] x_1_means = new double[n]; //1 относится к тому, что массив получится одномерный
    public static final double[] xy_1_means = new double[n];
    public static final double[][] x_2_means = new double[n][n];  // 2 относится к тому, что массив получится двумерный. Получается матричка размера n на n.
    public static double y_means = 0;
    public static final double[][] main_matrix = new double[n + 1][n + 1];
    public static final double[] right_matrix = new double[n + 1];

    public static void main(String[] args) {
        double[] yes = new double[N]; //Один длинный массив длины N
        double[][] xes = new double[N][n]; //N массивов по n в длину
        for (int i = 0; i < N; i++) {
            yes[i] = (Math.random() * 1);
        }
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < n; j++){
                xes[i][j] = (Math.random() * 1);
            }
        }
        System.out.println(Arrays.deepToString(xes));
        System.out.println("--------------");
        System.out.println(Arrays.toString(yes));
        System.out.println("---------------");


        y_means = Arrays.stream(yes).sum() / N; //средний от всех y

        for (int i = 0; i < n; i++) {
            x_1_means[i] = Arrays.stream(transposedMatrix(xes)[i]).sum() / N;
        } //считаем среднее от всех координат x по отдельности. <x_i>

        for (int i_s = 0; i_s < n; i_s++) {
            xy_1_means[i_s] = Arrays.stream(multiplication(yes, transposedMatrix(xes)[i_s])).sum() / N;
        } // <yx>

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                x_2_means[i][j] = Arrays.stream(multiplication(transposedMatrix(xes)[i],
                        transposedMatrix(xes)[j])).sum() / N;
            } // <x_ix_j>
        }
        // n+1 поскольку у нас 1ое уравнение есть и все остальные, которых n
        //для нахождения a_0 и a_i нужно будет обращать main_matrix
        for (int i_s = 0; i_s < n + 1; i_s++) {
            for (int i = 0; i < n + 1; i++) {
                main_matrix[i_s][i] = matrix_function(i, i_s);
            }
        }
        //напишем же правую часть уравнения
        for (int i_s = 0; i_s < n + 1; i_s++) {
            right_matrix[i_s] = right_part_function(i_s);
        }
        //Это строчка вида [a_0, a_1, a_2, ...]
        RealMatrix a = new Array2DRowRealMatrix(main_matrix);
        RealVector b = new ArrayRealVector(right_matrix);
        DecompositionSolver solver = new LUDecompositionImpl(a).getSolver();
        RealVector solution = solver.solve(b);
        System.out.println(solution);
    }

    public static double matrix_function(int i, int i_s) {
        if (i_s == 0) {
            //мы попали в самое первое уравнение
            if (i == 0) {
                return 1;
            } else return x_1_means[i - 1];
        } else {
            // мы попали в более сложный случай
            if (i == 0) {
                return x_1_means[i_s - 1];
            } else return x_2_means[i_s - 1][i - 1];
        }
    }

    public static double right_part_function(int i_s) {
        if (i_s == 0) return y_means;
        else return xy_1_means[i_s - 1];
    }

    public static double[][] transposedMatrix(double[][] sourceMatrix) {
        double[][] transposedMatrix = new double[n][N];
        for (int i = 0; i < sourceMatrix.length; i++) {
            for (int j = 0; j < sourceMatrix[i].length; j++) {
                transposedMatrix[j][i] = sourceMatrix[i][j];
            }
        }
        return transposedMatrix;
    }

    public static double[] multiplication(double[] matrix1, double[] matrix2) {
        double[] matrix3 = new double[matrix1.length];
        for (int i = 0; i < matrix1.length; i++) {
            matrix3[i] = matrix1[i]*matrix2[i];
        }
        return matrix3;
    }
}
