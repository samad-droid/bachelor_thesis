#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <Eigen/Dense>
#include <limits>

using namespace std;
using namespace Eigen;

// =======================================================
// Utility: Read the last column of a CSV file as labels
// =======================================================
inline vector<int> readLabelsFromCSV(const string &filename) {
    vector<int> labels;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "❌ Error: Cannot open " << filename << endl;
        return labels;
    }

    string line;
    bool header_skipped = false;
    while (getline(file, line)) {
        if (!header_skipped) { // skip header
            header_skipped = true;
            continue;
        }

        stringstream ss(line);
        string cell;
        vector<string> tokens;
        while (getline(ss, cell, ',')) tokens.push_back(cell);

        if (!tokens.empty()) {
            try {
                int label = stoi(tokens.back());
                labels.push_back(label);
            } catch (...) {
                cerr << "⚠️ Warning: Invalid label in line: " << line << endl;
            }
        }
    }
    return labels;
}

// =======================================================
// Utility: Hungarian matching (maximize total matches)
// =======================================================
inline vector<int> hungarianMatch(const MatrixXi &confusion) {
    const int n = confusion.rows();
    const int m = confusion.cols();
    const int dim = max(n, m);

    MatrixXd cost = MatrixXd::Zero(dim, dim);
    cost.setConstant(0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            cost(i, j) = -confusion(i, j); // negative = maximize matches

    vector<int> u(dim + 1), v(dim + 1), p(dim + 1), way(dim + 1);
    for (int i = 1; i <= dim; ++i) {
        p[0] = i;
        int j0 = 0;
        vector<double> minv(dim + 1, numeric_limits<double>::infinity());
        vector<char> used(dim + 1, false);
        do {
            used[j0] = true;
            int i0 = p[j0], j1 = 0;
            double delta = numeric_limits<double>::infinity();
            for (int j = 1; j <= dim; ++j) {
                if (used[j]) continue;
                double cur = cost(i0 - 1, j - 1) - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            for (int j = 0; j <= dim; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else minv[j] -= delta;
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }

    vector<int> match(n, -1);
    for (int j = 1; j <= dim; ++j)
        if (p[j] <= n && j <= m)
            match[p[j] - 1] = j - 1;
    return match;
}

// =======================================================
// Compute accuracy, per-cluster precision/recall/F1
// =======================================================
inline double computeClusteringMetrics(const string &trueFile, const string &predFile) {
    vector<int> trueLabels = readLabelsFromCSV(trueFile);
    vector<int> predLabels = readLabelsFromCSV(predFile);

    if (trueLabels.empty() || predLabels.empty() || trueLabels.size() != predLabels.size()) {
        cerr << "❌ Label files missing or size mismatch.\n";
        return 0.0;
    }

    unordered_map<int,int> trueMap, predMap;
    int tCount = 0, pCount = 0;
    for (int lbl : trueLabels)
        if (trueMap.find(lbl) == trueMap.end()) trueMap[lbl] = tCount++;
    for (int lbl : predLabels)
        if (predMap.find(lbl) == predMap.end()) predMap[lbl] = pCount++;

    MatrixXi confusion = MatrixXi::Zero(tCount, pCount);
    for (size_t i = 0; i < trueLabels.size(); ++i)
        confusion(trueMap[trueLabels[i]], predMap[predLabels[i]])++;

    vector<int> match = hungarianMatch(confusion);
    int correct = 0;
    for (int i = 0; i < tCount; ++i)
        if (match[i] != -1)
            correct += confusion(i, match[i]);
    double accuracy = 100.0 * correct / trueLabels.size();

    cout << "\n===== Clustering Evaluation =====\n";
    cout << "Points: " << trueLabels.size()
         << " | True clusters: " << tCount
         << " | Pred clusters: " << pCount << "\n";
    cout << "Global accuracy: " << accuracy << " %\n";

    double totalPrec=0,totalRec=0,totalF1=0; int used=0;
    cout << "\nCluster | Precision | Recall | F1\n";
    cout << "-----------------------------------\n";
    for (int i = 0; i < tCount; ++i) {
        if (match[i] == -1) continue;
        int j = match[i];
        double tp = confusion(i,j);
        double fp = confusion.col(j).sum() - tp;
        double fn = confusion.row(i).sum() - tp;
        double prec = (tp+fp>0)?tp/(tp+fp):0;
        double rec  = (tp+fn>0)?tp/(tp+fn):0;
        double f1   = (prec+rec>0)?2*prec*rec/(prec+rec):0;
        cout << i << " ↔ " << j << " | " << prec << " | " << rec << " | " << f1 << "\n";
        totalPrec+=prec; totalRec+=rec; totalF1+=f1; used++;
    }
    if (used>0) {
        cout << "\nMacro Precision: " << totalPrec/used
             << " | Recall: " << totalRec/used
             << " | F1: " << totalF1/used << "\n";
    }
    cout << "===================================\n";
    return accuracy;
}
