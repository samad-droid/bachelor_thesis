#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <set>

// Utility function to split a CSV line
static std::vector<std::string> splitCSVLine(const std::string &line) {
    std::vector<std::string> result;
    std::stringstream ss(line);
    std::string item;
    bool insideQuote = false;
    std::string current;

    for (char c : line) {
        if (c == '"') {
            insideQuote = !insideQuote;
        } else if (c == ',' && !insideQuote) {
            result.push_back(current);
            current.clear();
        } else {
            current += c;
        }
    }
    result.push_back(current);
    return result;
}

// Main function to merge CSV files
static void mergeDetectedAndMeanQDF(
    const std::string &detectedFile,
    const std::string &meanFile,
    const std::string &outputFile)
{
    std::ifstream detected(detectedFile);
    std::ifstream mean(meanFile);
    if (!detected.is_open() || !mean.is_open()) {
        std::cerr << "Error opening input files.\n";
        return;
    }

    std::unordered_map<int, std::vector<std::string>> detectedRows;
    std::string line;
    std::getline(detected, line); // header
    std::string detectedHeader = line;

    // Count how many times each cluster_id appears
    while (std::getline(detected, line)) {
        auto tokens = splitCSVLine(line);
        if (tokens.empty()) continue;
        int cluster_id = std::stoi(tokens[0]);
        detectedRows[cluster_id].push_back(line);
    }

    // Find cluster_ids with exactly one line
    std::set<int> uniqueClusters;
    for (const auto &entry : detectedRows) {
        if (entry.second.size() == 1)
            uniqueClusters.insert(entry.first);
    }

    // Collect lines to write
    std::vector<std::string> outputLines;

    // Read all from mean_qdf_lines.csv (keep header)
    std::string meanHeader;
    std::getline(mean, meanHeader);
    outputLines.push_back(meanHeader);

    while (std::getline(mean, line)) {
        outputLines.push_back(line);
    }

    // Append the unique cluster lines from detected_subspaces.csv
    for (int id : uniqueClusters) {
        for (const auto &row : detectedRows[id]) {
            outputLines.push_back(row);
        }
    }

    // Write to output file
    std::ofstream out(outputFile);
    if (!out.is_open()) {
        std::cerr << "Error opening output file.\n";
        return;
    }

    for (const auto &l : outputLines)
        out << l << "\n";

    std::cout << "✅ Merged file written to: " << outputFile << "\n";
}
