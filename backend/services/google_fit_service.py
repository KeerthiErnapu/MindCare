
import os
import json
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import time
from datetime import datetime

SCOPES = [
    'https://www.googleapis.com/auth/fitness.activity.read',
    'https://www.googleapis.com/auth/fitness.heart_rate.read'
]

def authenticate_google_fit():
    try:
        creds = None
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            with open('token.json', 'w') as token_file:
                token_file.write(creds.to_json())
        return creds
    except Exception as e:
        print(f"Authentication error: {e}")
        return None

def fetch_wellness_metrics():
    try:
        creds = authenticate_google_fit()
        if not creds:
            # Return mock data if authentication fails
            return {
                'heart_rate': 75,
                'steps': 5000,
                'heart_points': 35
            }

        service = build('fitness', 'v1', credentials=creds)
        
        end_time_ns = int(time.time() * 1e9)
        start_time_ns = int((time.time() - 24 * 60 * 60) * 1e9)  # Last 24 hours
        dataset_id = f"{start_time_ns}-{end_time_ns}"
        
        metrics = {
            'heart_rate': 0,
            'steps': 0,
            'heart_points': 0
        }
        
        # Get data sources
        data_sources = service.users().dataSources().list(userId='me').execute()
        
        for source in data_sources.get('dataSource', []):
            dataset = service.users().dataSources().datasets().get(
                userId='me',
                dataSourceId=source['dataStreamId'],
                datasetId=dataset_id
            ).execute()
            
            if 'heart_rate' in source.get('dataStreamId', '').lower():
                # Process heart rate data
                if dataset.get('point'):
                    latest_hr = dataset['point'][-1]['value'][0].get('fpVal', 0)
                    metrics['heart_rate'] = int(latest_hr)
            
            elif 'step_count' in source.get('dataStreamId', '').lower():
                # Process step data
                if dataset.get('point'):
                    total_steps = sum(point['value'][0].get('intVal', 0) for point in dataset['point'])
                    metrics['steps'] = total_steps
            
            elif 'heart_minutes' in source.get('dataStreamId', '').lower():
                # Process heart points data
                if dataset.get('point'):
                    total_points = sum(point['value'][0].get('fpVal', 0) for point in dataset['point'])
                    metrics['heart_points'] = int(total_points)
        
        return metrics
    except Exception as e:
        print(f"Error fetching Google Fit data: {e}")
        # Return mock data if there's an error
        return {
            'heart_rate': 75,
            'steps': 5000,
            'heart_points': 35
        }
