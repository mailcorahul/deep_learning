import json
import pandas as pd

# method to return the ProductUrlIds for all PIDs in ProductFamily
def get_prod_urls(productFamily, prodUrl2pid):

	prodUrls = set();
	for pid in productFamily:
		for prodUrl, pids in prodUrl2pid.items():
			if pid in pids:
				prodUrls.add(prodUrl);

	return list(prodUrls);

# loading producturl2pid map
with open('../data/TOPS_produrl2pid.json') as f:
	prodUrl2pid = json.load(f);
print('Number of unique TOPS products {}'.format(len(prodUrl2pid)));

# loading pid2producturl map
pid2prodUrl = pd.read_csv('../data/TOPS_pid2produrl.csv');
print('Number of TOPS PIDs {}'.format(len(pid2prodUrl)));
