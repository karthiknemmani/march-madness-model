import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import time

"""
Get Kenpom data and export to csv
"""

headers = {
    "User-Agent":
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.79 Safari/537.36"
}

dfs = []

for season in range(2003, 2025):
    url = f"https://kenpom.com/index.php?y={season}"
    with requests.Session() as request:
        response = request.get(url, timeout=30,  headers=headers)
    if response.status_code != 200:
        print(response.raise_for_status())

    soup = BeautifulSoup(response.text, "html.parser")

    table_full = soup.find_all('table', {'id': 'ratings-table'})
    
    # Assuming there's only one thead per table as per usual table structure
    thead = table_full[0].find('thead')
    column_headers = [header.text for header in thead.find_all('th')]
    
    table = str(table_full[0]).replace(str(thead), '')
    df = pd.read_html(table)[0]
    
    # Assign extracted headers to the DataFrame
    df.columns = column_headers
    
    df['Season'] = season
    
    dfs.append(df)
    time.sleep(2)
    

col_vals = ['Rk', 'Team', 'Conf', 'W-L', 'AdjEM', 'AdjO', 'AdjORk', 'AdjD', 'AdjDRk', 'AdjT', 'AdjTRk', 'Luck', 'LuckRk', 'AdjSOS', 'AdjSOSRk', 'OppO', 'OppORk', 'OppD', 'OppDRk', 'NCSOS', 'NCSOSRk', 'Season']
full_df = pd.concat(dfs, ignore_index=True)

full_df.columns = col_vals

full_df['Team'] = full_df['Team'].apply(lambda x: re.sub(r'\d+', '', x).strip())
full_df['Team'] = full_df['Team'].apply(lambda x: re.sub(r'\.', '', x).strip())
full_df['Team'] = full_df['Team'].apply(lambda x: re.sub(r'Saint', 'St', x).strip())
full_df['Team'] = full_df['Team'].apply(lambda x: re.sub(r'Mount', 'Mt', x).strip())
full_df['Team'] = full_df['Team'].apply(lambda x: re.sub(r'Western', 'W', x).strip())
full_df['Team'] = full_df['Team'].apply(lambda x: re.sub(r'Eastern', 'E', x).strip())
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Florida Atlantic', 'FL Atlantic'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('UT Rio Grande Valley', 'UTRGV'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Texas A&M Commerce', 'TX A&M Commerce'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Texas A&M Corpus Chris', 'TAM C. Christi'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Kent St', 'Kent'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Central Connecticut', 'Central Conn'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('''St Joseph's''', '''St Joseph's PA'''))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Milwaukee', 'WI Milwaukee'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Southern Illinois', 'S Illinois'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Texas Southern', 'TX Southern'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('W Kentucky', 'WKU'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Illinois Chicago', 'IL Chicago'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Louisiana Lafayette', 'Louisiana'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Central Michigan', 'C Michigan'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Southwest Missouri St', 'Missouri St'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Troy St', 'Troy'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('George Washington', 'G Washington'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Loyola Chicago', 'Loyola-Chicago'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Boston University', 'Boston Univ'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Middle Tennessee St', 'MTSU'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Northern Illinois', 'N Illinois'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Stephen F Austin', 'SF Austin'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('American', 'American Univ'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('East Tennessee St', 'ETSU'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Cal St Northridge', 'CS Northridge'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Southwest Texas St', 'Texas St'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Green Bay', 'WI Green Bay'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Loyola Marymount', 'Loy Marymount'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('FIU', 'Florida Intl'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Birmingham Southern', 'Birmingham So'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Tennessee Martin', 'TN Martin'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Georgia Southern', 'Ga Southern'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Cal St Fullerton', 'CS Fullerton'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Mississippi Valley St', 'MS Valley St'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Monmouth', 'Monmouth NJ'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('South Carolina St', 'S Carolina St'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Sacramento St', 'CS Sacramento'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Prairie View A&M', 'Prairie View'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Charleston', 'Col Charleston'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('College of Col Charleston', 'Col Charleston'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Col Charleston Southern', 'Charleston So'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Southeast Missouri St', 'SE Missouri St'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Louisiana Monroe', 'ULM'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Texas Southern', 'TX Southern'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Coastal Carolina', 'Coastal Car'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Southeastern Louisiana', 'SE Louisiana'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('UMKC', 'Missouri KC'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Fairleigh Dickinson', 'F Dickinson'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('UTSA', 'UT San Antonio'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('The Citadel', 'Citadel'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Grambling St', 'Grambling'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Northwestern St', 'Northwestern LA'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Albany', 'SUNY Albany'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Bethune Cookman', 'Bethune-Cookman'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Maryland E Shore', 'MD E Shore'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Arkansas Pine Bluff', 'Ark Pine Bluff'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('North Carolina A&T', 'NC A&T'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Middle Tennessee', 'MTSU'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Utah Valley St', 'Utah Valley'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Northern Colorado', 'N Colorado'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('North Dakota St', 'N Dakota St'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('South Dakota St', 'S Dakota St'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Kennesaw St', 'Kennesaw'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Central Arkansas', 'Cent Arkansas'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Winston Salem St', 'W Salem St'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('USC Upstate', 'SC Upstate'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Florida Gulf Coast', 'FL Gulf Coast'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Cal St Bakersfield', 'CS Bakersfield'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('North Carolina Central', 'NC Central'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('SIU Edwardsville', 'SIUE'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Houston Baptist', 'Houston Chr'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Nebraska Omaha', 'NE Omaha'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Northern Kentucky', 'N Kentucky'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('UMass Lowell', 'MA Lowell'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Abilene Christian', 'Abilene Chr'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Fort Wayne', 'PFW'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Little Rock', 'Ark Little Rock'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Arkansas Ark Little Rock', 'Ark Little Rock'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Purdue PFW', 'PFW'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('IPFW', 'PFW'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('LIU', 'LIU Brooklyn'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('LIU Brooklyn Brooklyn', 'LIU Brooklyn'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Dixie St', 'Utah Tech'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Detroit Mercy', 'Detroit'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('St Thomas', 'St Thomas MN'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Queens', 'Queens NC'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Houston Christian', 'Houston Chr'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('St Francis', 'St Francis PA'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('St Francis PA PA', 'St Francis PA'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('St Francis PA NY', 'St Francis NY'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('St Mary\'s', 'St Mary\'s CA'))
full_df['Team'] = full_df['Team'].apply(lambda x: x.replace('Mt St Mary\'s CA', 'Mt St Mary\'s'))

# if there is a team called Southern, change it to Southern Univ
full_df['Team'] = full_df['Team'].apply(lambda x: 'Southern Univ' if x == 'Southern' else x)

# remove texas pan american and texas pan american univ
full_df = full_df[~full_df['Team'].isin(['Texas Pan American', 'Texas Pan American Univ'])]
full_df = full_df[full_df['Team'] != 'Texas Pan American']

# final kenpom df
kenpom = full_df.copy()
kenpom = kenpom[kenpom['Season'] != 2020]
kenpom = kenpom.reset_index(drop=True)
kenpom.to_csv('../../data/MKenpomData.csv', index=False)
