import pandas as pd
import logging
import re
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def similar(a, b, threshold=0.8):
    """Calculate similarity ratio between two strings.

    Args:
        a (str): First string.
        b (str): Second string.
        threshold (float): Minimum similarity score to consider a match.

    Returns:
        bool: True if similarity ratio exceeds threshold, False otherwise.
    """
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio() > threshold

def clean_number(value):
    """Convert a string to a float, handling various number formats including negative values in parentheses.

    Args:
        value: Input value to convert.

    Returns:
        float or None: Converted float value or None if conversion fails.
    """
    try:
        value_str = str(value).strip()
        # Check if the value is enclosed in parentheses, indicating a negative number
        if value_str.startswith('(') and value_str.endswith(')'):
            # Remove parentheses and negate the number
            value_str = '-' + value_str[1:-1].strip()
        # Clean non-numeric characters except for minus and decimal point
        cleaned = re.sub(r'[^\d\-.]', '', value_str).replace(',', '.')
        return float(cleaned)
    except (ValueError, AttributeError):
        return None

def select_key_for_year(df, year, key_column='Key'):
    """Extract financial metrics from a DataFrame for a specific year.

    Args:
        df (pd.DataFrame): DataFrame with a key column and year columns.
        year (str): Year to extract data for, or None if no year is available.
        key_column (str): Name of the column containing metric keys (default: 'Key').

    Returns:
        dict: Dictionary mapping financial metrics to their values.

    Raises:
        KeyError: If key_column is not in DataFrame columns.
    """
    if key_column not in df.columns:
        logging.error(f"Key column '{key_column}' not found in DataFrame. Available columns: {list(df.columns)}")
        raise KeyError(f"Key column '{key_column}' not found in DataFrame")

    results = {
        'actifs_courants': None,
        'passifs_courants': None,
        'stocks': None,
        'resultat_net': None,
        'revenus': None,
        'total_actifs': None,
        'capitaux_propres': None,
        'total_passifs': None,
        'capital_social': None,
        'reserves': None,
        'resultat_reporte': None,
        'autres_capitaux_propres': None,
        'actions_propres': None
    }

    for index, row in df.iterrows():
        key = str(row[key_column]).lower()
        value = clean_number(row[year]) if year else None
        if value is None:
            continue

        if similar(key, 'total des actifs courants'):
            results['actifs_courants'] = value
        elif similar(key, 'total des passifs courants'):
            results['passifs_courants'] = value
        elif similar(key, 'stocks'):
            results['stocks'] = value
        elif similar(key, "Résultat de l'exercice"):
            results['resultat_net'] = value
        elif similar(key, 'revenus'):
            results['revenus'] = value
        elif similar(key, 'total des actifs'):
            results['total_actifs'] = value
        elif similar(key, 'capitaux propres'):
            results['capitaux_propres'] = value
        elif similar(key, 'total des passifs'):
            results['total_passifs'] = value
        elif similar(key, 'capital social'):
            results['capital_social'] = value
        elif similar(key, 'réserves'):
            results['reserves'] = value
        elif similar(key, 'résultat reporté'):
            results['resultat_reporte'] = value
        elif similar(key, 'autres capitaux propres'):
            results['autres_capitaux_propres'] = value
        elif similar(key, 'actions propres'):
            results['actions_propres'] = value

    return results

def select_key(df, key_column='Key'):
    """Extract financial metrics from a DataFrame for the first available year.

    Args:
        df (pd.DataFrame): DataFrame with a key column and year columns.
        key_column (str): Name of the column containing metric keys (default: 'Key').

    Returns:
        dict: Dictionary mapping financial metrics to their values.
    """
    year_columns = [col for col in df.columns if col not in [key_column, 'source_image']]
    if not year_columns:
        logging.warning("No year columns found in DataFrame")
        return select_key_for_year(df, None, key_column)

    return select_key_for_year(df, year_columns[0], key_column)

def calculate_ratios(df, key_column='Key'):
    """Calculate financial ratios for each year in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with a key column and year columns.
        key_column (str): Name of the column containing metric keys (default: 'Key').

    Returns:
        dict: Dictionary of ratios for each year, or empty dict if calculation fails.
    """
    try:
        if df.empty:
            logging.warning("Empty DataFrame provided to calculate_ratios")
            return {}

        logging.info(f"DataFrame columns: {list(df.columns)}")
        logging.info(f"DataFrame shape: {df.shape}")
        logging.info(f"DataFrame sample:\n{df.head().to_string()}")

        if key_column not in df.columns:
            logging.error(f"Key column '{key_column}' not found in DataFrame. Available columns: {list(df.columns)}")
            return {}

        ratios = {}
        year_columns = [col for col in df.columns if col not in [key_column, 'source_image']]

        if not year_columns:
            logging.warning("No valid year columns found in DataFrame")
            return {}

        for year in year_columns:
            year_ratios = {}
            results = select_key_for_year(df, year, key_column)

            if results['actifs_courants'] and results['passifs_courants'] and results['passifs_courants'] != 0:
                year_ratios['Ratio liquidité générale'] = results['actifs_courants'] / results['passifs_courants']
            if results['actifs_courants'] and results['stocks'] and results['passifs_courants'] and results['passifs_courants'] != 0:
                year_ratios['Ratio de liquidité immédiate'] = (results['actifs_courants'] - results['stocks']) / results['passifs_courants']
            if results['resultat_net'] and results['revenus'] and results['revenus'] != 0:
                year_ratios['Marge nette'] = results['resultat_net'] / results['revenus']
            if results['resultat_net'] and results['total_actifs'] and results['total_actifs'] != 0:
                year_ratios['Rentabilité économique'] = results['resultat_net'] / results['total_actifs']
            if results['resultat_net'] and results['capitaux_propres'] and results['capitaux_propres'] != 0:
                year_ratios['Rentabilité financière'] = results['resultat_net'] / results['capitaux_propres']
            if results['total_passifs'] and results['capitaux_propres'] and results['capitaux_propres'] != 0:
                year_ratios['Ratio d\'endettement'] = results['total_passifs'] / results['capitaux_propres']
            if results['capitaux_propres'] and results['total_actifs'] and results['total_actifs'] != 0:
                year_ratios['Ratio de solvabilité'] = results['capitaux_propres'] / results['total_actifs']

            ratios[year] = year_ratios

        return ratios

    except (ValueError, KeyError) as e:
        logging.error(f"Error calculating ratios: {str(e)}")
        return {}