import logging
import requests
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WARDataFetcher:
    """Fetch WAR data from Baseball-Reference, FanGraphs, and Lahman database"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (MLB Research Project)'
        })
    
    def fetch_baseball_reference_war(self, start_year: int = 1985) -> pd.DataFrame:
        """
        Fetch WAR data from Baseball-Reference
        
        Args:
            start_year: Starting year for data collection
            
        Returns:
            DataFrame with WAR data
        """
        logger.info("Fetching Baseball-Reference WAR data...")
        
        try:
            # Baseball-Reference provides historical WAR data
            url = "https://www.baseball-reference.com/data/war_daily_bat.txt"
            
            df = pd.read_csv(url)
            
            # Filter by year
            df = df[df['year_ID'] >= start_year].copy()
            
            # Standardize column names
            df = df.rename(columns={
                'year_ID': 'year',
                'player_ID': 'player_id',
                'age': 'age',
                'WAR': 'war',
                'G': 'games_played'
            })
            
            logger.info(f"Fetched {len(df)} player-season records from Baseball-Reference")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Baseball-Reference data: {e}")
            raise
    
    def fetch_fangraphs_war(self, start_year: int = 1985, end_year: int = None) -> pd.DataFrame:
        """
        Fetch WAR data from FanGraphs
        
        Args:
            start_year: Starting year
            end_year: Ending year (defaults to current year)
            
        Returns:
            DataFrame with FanGraphs WAR data
        """
        if end_year is None:
            end_year = datetime.now().year
        
        logger.info(f"Fetching FanGraphs WAR data ({start_year}-{end_year})...")
        
        all_data = []
        
        for year in range(start_year, end_year + 1):
            try:
                # FanGraphs leaderboard API (batting)
                url = "https://www.fangraphs.com/api/leaders/major-league/data"
                params = {
                    'pos': 'all',
                    'stats': 'bat',
                    'lg': 'all',
                    'qual': '0',  # 0 = all players
                    'season': year,
                    'season1': year,
                    'startdate': f'{year}-01-01',
                    'enddate': f'{year}-12-31',
                    'type': '8',  # Standard batting
                    'pageitems': '10000000'
                }
                
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                if 'data' in data:
                    df_year = pd.DataFrame(data['data'])
                    df_year['year'] = year
                    all_data.append(df_year)
                
                time.sleep(1)  # Rate limiting
                logger.info(f"Fetched {year} data")
                
            except Exception as e:
                logger.warning(f"Error fetching FanGraphs {year}: {e}")
                continue
        
        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            
            # Standardize columns
            df = df.rename(columns={
                'Season': 'year',
                'playerid': 'player_id',
                'Name': 'name',
                'Age': 'age',
                'WAR': 'war',
                'G': 'games_played'
            })
            
            logger.info(f"Fetched {len(df)} player-season records from FanGraphs")
            return df
        else:
            logger.warning("No FanGraphs data fetched")
            return pd.DataFrame()
    
    def fetch_lahman_database(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch Lahman database (historical baseball statistics)
        
        Returns:
            Dictionary of DataFrames (Batting, Pitching, People, Salaries)
        """
        logger.info("Fetching Lahman database...")
        
        base_url = "https://raw.githubusercontent.com/chadwickbureau/baseballdatabank/master/core/"
        
        tables = {
            'batting': 'Batting.csv',
            'pitching': 'Pitching.csv',
            'people': 'People.csv',
            'salaries': 'Salaries.csv',
            'appearances': 'Appearances.csv'
        }
        
        data = {}
        
        for name, file in tables.items():
            try:
                url = base_url + file
                df = pd.read_csv(url)
                data[name] = df
                logger.info(f"Fetched {name}: {len(df)} records")
            except Exception as e:
                logger.error(f"Error fetching {name}: {e}")
        
        return data
    
    def merge_war_sources(self, bref_df: pd.DataFrame, fg_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge WAR data from multiple sources
        
        Args:
            bref_df: Baseball-Reference WAR data
            fg_df: FanGraphs WAR data
            
        Returns:
            Combined DataFrame with both WAR metrics
        """
        logger.info("Merging WAR sources...")
        
        # Standardize player IDs and merge
        if not fg_df.empty:
            merged = pd.merge(
                bref_df,
                fg_df[['year', 'player_id', 'war']],
                on=['year', 'player_id'],
                how='outer',
                suffixes=('_bref', '_fg')
            )
        else:
            merged = bref_df.copy()
            merged['war_bref'] = merged['war']
            merged['war_fg'] = None
        
        # Use average WAR when both available
        merged['war'] = merged[['war_bref', 'war_fg']].mean(axis=1, skipna=True)
        
        logger.info(f"Merged dataset: {len(merged)} records")
        return merged
    
    def get_player_positions(self, lahman_data: Dict[str, pd.DataFrame], 
                            year: int) -> pd.DataFrame:
        """
        Extract primary positions for players in a given year
        
        Args:
            lahman_data: Lahman database tables
            year: Season year
            
        Returns:
            DataFrame with player positions
        """
        appearances = lahman_data.get('appearances')
        if appearances is None:
            return pd.DataFrame()
        
        # Filter by year
        year_apps = appearances[appearances['yearID'] == year].copy()
        
        # Position columns
        pos_cols = ['G_p', 'G_c', 'G_1b', 'G_2b', 'G_3b', 'G_ss', 
                   'G_lf', 'G_cf', 'G_rf', 'G_dh']
        pos_map = {
            'G_p': 'P', 'G_c': 'C', 'G_1b': '1B', 'G_2b': '2B',
            'G_3b': '3B', 'G_ss': 'SS', 'G_lf': 'LF', 'G_cf': 'CF',
            'G_rf': 'RF', 'G_dh': 'DH'
        }
        
        # Find primary position (most games)
        def get_primary_pos(row):
            pos_games = {pos_map[col]: row[col] for col in pos_cols if col in row}
            if not pos_games or all(pd.isna(v) or v == 0 for v in pos_games.values()):
                return None
            return max(pos_games.items(), key=lambda x: x[1] if not pd.isna(x[1]) else 0)[0]
        
        year_apps['position'] = year_apps.apply(get_primary_pos, axis=1)
        
        return year_apps[['playerID', 'yearID', 'position']].rename(
            columns={'playerID': 'player_id', 'yearID': 'year'}
        )
    
    def save_data(self, df: pd.DataFrame, filename: str):
        """Save DataFrame to CSV and Parquet"""
        csv_path = self.data_dir / f"{filename}.csv"
        parquet_path = self.data_dir / f"{filename}.parquet"
        
        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False)
        
        logger.info(f"Saved {filename}: {len(df)} records")


def main():
    """Example usage"""
    from pathlib import Path
    
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    fetcher = WARDataFetcher(data_dir)
    
    # Fetch data
    bref_war = fetcher.fetch_baseball_reference_war(start_year=1985)
    lahman_data = fetcher.fetch_lahman_database()
    
    # Save
    fetcher.save_data(bref_war, "war_baseball_reference")
    
    for name, df in lahman_data.items():
        fetcher.save_data(df, f"lahman_{name}")


if __name__ == "__main__":
    main()
