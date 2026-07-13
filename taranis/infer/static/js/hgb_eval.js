// Evaluate an HGB tree bundle exported by scripts/export_hgb_json.py.
//
// The bundle is a plain object:
//   { base_score, trees: [ { feature, threshold, left, right, value }, ... ] }
//
// For each tree, walk from node 0. If feature[node] >= 0, compare
// features[feature[node]] <= threshold[node]; go left on true, right on false.
// When a leaf is reached (feature[node] = -1), accumulate value[node].
// Sum contributions from all trees, add base_score, apply logistic sigmoid.

function walkTree(tree, features) {
    let node = 0;
    while (tree.feature[node] >= 0) {
        const f = tree.feature[node];
        node = features[f] <= tree.threshold[node] ? tree.left[node] : tree.right[node];
    }
    return tree.value[node];
}

export function predictHGB(bundle, features) {
    let score = bundle.base_score;
    for (const tree of bundle.trees) {
        score += walkTree(tree, features);
    }
    return 1.0 / (1.0 + Math.exp(-score));
}

export function alertFromProba(bundle, proba) {
    if (proba >= bundle.rouge_threshold) return "ROUGE";
    if (proba >= bundle.orange_threshold) return "ORANGE";
    return "VERT";
}
