import logging
import requests
import pandas as pd
from pathlib import Path
from typing import Optional, List
from bs4 import BeautifulSoup
import time
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContractDataFetcher:
    """Fetch contract data from Spotrac and other sources"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (MLB Research Project)'
        })
    
    def fetch_spotrac_contracts(self, year: int) -> pd.DataFrame:
        """
        Fetch contract data from Spotrac for a given year
        
        Args:
            year: Season year
            
        Returns:
            DataFrame with contract information
        """
        logger.info(f"Fetching Spotrac contracts for {year}...")
        
        try:
            # Spotrac historical contracts page
            url = f"https://www.spotrac.com/mlb/contracts/{year}/"
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Parse contract table
            contracts = []
            table = soup.find('table', {'class': 'datatable'})
            
            if table:
                rows = table.find_all('tr')[1:]  # Skip header
                
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 5:
                        player_name = cols[0].text.strip()
                        team = cols[1].text.strip() if len(cols) > 1 else None
                        contract_value = self._parse_money(cols[2].text.strip())
                        contract_length = self._parse_years(cols[3].text.strip())
                        contract_type = self._determine_contract_type(cols[4].text.strip())
                        
                        contracts.append({
                            'name': player_name,
                            'team': team,
                            'year': year,
                            'total_value': contract_value,
                            'contract_length': contract_length,
                            'contract_type': contract_type,
                            'salary': contract_value / contract_length if contract_length > 0 else contract_value
                        })
            
            df = pd.DataFrame(contracts)
            logger.info(f"Fetched {len(df)} contracts for {year}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Spotrac data for {year}: {e}")
            return pd.DataFrame()
    
    def fetch_lahman_salaries(self, lahman_salaries: pd.DataFrame, 
                             start_year: int = 1985) -> pd.DataFrame:
        """
        Process Lahman salary data
        
        Args:
            lahman_salaries: Lahman Salaries DataFrame
            start_year: Starting year
            
        Returns:
            Processed salary DataFrame
        """
        logger.info("Processing Lahman salary data...")
        
        df = lahman_salaries[lahman_salaries['yearID'] >= start_year].copy()
        
        df = df.rename(columns={
            'playerID': 'player_id',
            'yearID': 'year',
            'teamID': 'team',
            'salary': 'salary'
        })
        
        # Estimate contract type based on salary and player service time
        # This is a simplified heuristic
        df['contract_type'] = df.apply(self._estimate_contract_type, axis=1)
        
        logger.info(f"Processed {len(df)} salary records")
        return df
    
    def fetch_arbitration_data(self, start_year: int = 1985, 
                              end_year: int = None) -> pd.DataFrame:
        """
        Fetch arbitration award data
        
        Note: This is a placeholder. Real implementation would need to:
        1. Scrape MLB Trade Rumors arbitration tracker
        2. Use MLBPA datasets if available
        3. Parse historical arbitration awards from news sources
        
        Args:
            start_year: Starting year
            end_year: Ending year
            
        Returns:
            DataFrame with arbitration data
        """
        if end_year is None:
            from datetime import datetime
            end_year = datetime.now().year
        
        logger.info(f"Fetching arbitration data ({start_year}-{end_year})...")
        
        # Placeholder for arbitration data
        # In production, this would scrape from MLB Trade Rumors or similar
        arb_data = []
        
        # Example structure
        arb_df = pd.DataFrame(arb_data, columns=[
            'player_id', 'name', 'year', 'salary', 
            'filed_amount', 'team_offer', 'settlement', 'hearing_result'
        ])
        
        logger.info(f"Arbitration data: {len(arb_df)} records")
        return arb_df
    
    def separate_contract_types(self, df: pd.DataFrame) -> tuple:
        """
        Separate contracts into arbitration and free agency
        
        Args:
            df: Combined contract DataFrame
            
        Returns:
            Tuple of (arbitration_df, free_agency_df)
        """
        logger.info("Separating contract types...")
        
        arb_df = df[df['contract_type'] == 'arbitration'].copy()
        fa_df = df[df['contract_type'] == 'free_agency'].copy()
        
        logger.info(f"Arbitration: {len(arb_df)}, Free Agency: {len(fa_df)}")
        
        return arb_df, fa_df
    
    def _parse_money(self, text: str) -> float:
        """Parse money string to float"""
        try:
            # Remove $ and commas, handle M/K suffixes
            text = text.replace('$', '').replace(',', '').strip()
            
            if 'M' in text:
                return float(text.replace('M', '')) * 1_000_000
            elif 'K' in text:
                return float(text.replace('K', '')) * 1_000
            else:
                return float(text)
        except:
            return 0.0
    
    def _parse_years(self, text: str) -> int:
        """Parse contract length"""
        try:
            match = re.search(r'(\d+)', text)
            return int(match.group(1)) if match else 1
        except:
            return 1
    
    def _determine_contract_type(self, text: str) -> str:
        """Determine if contract is arbitration or free agency"""
        text_lower = text.lower()
        
        if 'arb' in text_lower or 'arbitration' in text_lower:
            return 'arbitration'
        elif 'free' in text_lower or 'fa' in text_lower:
            return 'free_agency'
        else:
            # Default heuristic: 1-year deals often arbitration, multi-year often FA
            return 'arbitration'
    
    def _estimate_contract_type(self, row) -> str:
        """
        Estimate contract type from Lahman data
        
        Heuristic:
        - Players typically reach arbitration after 3 years
        - Free agency after 6 years
        - This is simplified and not perfectly accurate
        """
        # Would need service time data for accuracy
        # For now, use salary thresholds as rough proxy
        
        if row['salary'] < 1_000_000:
            return 'pre_arbitration'
        elif row['salary'] < 5_000_000:
            return 'arbitration'
        else:
            return 'free_agency'
    
    def enrich_with_service_time(self, contracts_df: pd.DataFrame, 
                                 lahman_batting: pd.DataFrame) -> pd.DataFrame:
        """
        Add service time estimates to contracts
        
        Args:
            contracts_df: Contract DataFrame
            lahman_batting: Lahman batting data for calculating service time
            
        Returns:
            Enriched DataFrame with service time
        """
        logger.info("Enriching contracts with service time...")
        
        # Calculate rookie year for each player
        rookie_years = lahman_batting.groupby('playerID')['yearID'].min().reset_index()
        rookie_years.columns = ['player_id', 'rookie_year']
        
        # Merge
        enriched = contracts_df.merge(rookie_years, on='player_id', how='left')
        enriched['service_years'] = enriched['year'] - enriched['rookie_year']
        
        # Refine contract type based on service time
        def refine_type(row):
            if pd.isna(row['service_years']):
                return row['contract_type']
            
            if row['service_years'] < 3:
                return 'pre_arbitration'
            elif row['service_years'] < 6:
                return 'arbitration'
            else:
                return 'free_agency'
        
        enriched['contract_type'] = enriched.apply(refine_type, axis=1)
        
        return enriched
    
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
    
    fetcher = ContractDataFetcher(data_dir)
    
    # Load Lahman salary data (assumes already fetched)
    try:
        lahman_salaries = pd.read_parquet(data_dir / "lahman_salaries.parquet")
        salary_df = fetcher.fetch_lahman_salaries(lahman_salaries)
        
        # Separate by type
        arb_df, fa_df = fetcher.separate_contract_types(salary_df)
        
        # Save
        fetcher.save_data(arb_df, "contracts_arbitration")
        fetcher.save_data(fa_df, "contracts_free_agency")
        
    except FileNotFoundError:
        logger.error("Lahman salary data not found. Run war_data_fetcher.py first.")


if __name__ == "__main__":
    main()
