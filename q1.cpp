#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

vector<long long> a;
vector<vector<int>> adj;
vector<long long> result;

void calculate(int u, int p, long long pluscon, long long negcon) {
    result[u] = a[u] + negcon;

    long long currentp = max(0LL, a[u] + negcon);
    long long currentn = max(0LL, -a[u] + pluscon);

    for (int v : adj[u]) {
        if (v == p) {
            continue;
        }
        calculate(v, u, currentp, currentn);
    }
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int t_cases;
    cin >> t_cases;
    while (t_cases--) {
        int n_vertices;
        cin >> n_vertices;

        a.assign(n_vertices, 0);
        for (int i = 0; i < n_vertices; ++i) {
            cin >> a[i];
        }

        adj.assign(n_vertices, vector<int>());
        for (int i = 0; i < n_vertices - 1; ++i) {
            int u, v;
            cin >> u >> v;
            --u;
            --v;
            adj[u].push_back(v);
            adj[v].push_back(u);
        }

        result.assign(n_vertices, 0);

        calculate(0, -1, 0LL, 0LL);

        for (int i = 0; i < n_vertices; ++i) {
            cout << result[i] << (i == n_vertices - 1 ? "" : " ");
        }
        cout << endl;
    }
    return 0;
}
