import json
attack_map = {
    'BENIGN': 'BENIGN',
    'DDoS': 'DDoS',
    'DoS Hulk': 'DoS',
    'DoS GoldenEye': 'DoS',
    'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'DoS',
    'PortScan': 'Port Scan',
    'FTP-Patator': 'Brute Force',
    'SSH-Patator': 'Brute Force',
    'Bot': 'Bot',
    'Web Attack � Brute Force': 'Web Attack',
    'Web Attack � XSS': 'Web Attack',
    'Web Attack � Sql Injection': 'Web Attack',
    'Infiltration': 'Infiltration',
    'Heartbleed': 'Heartbleed'
}

features_target_classes = [
    'BENIGN', 'DDoS', 'PortScan', 'Bot', 'Infiltration',
    'Web Attack – Brute Force', 'Web Attack – XSS',
    'Web Attack – Sql Injection', 'FTP-Patator', 'SSH-Patator',
    'DoS slowloris', 'DoS Slowhttptest', 'DoS Hulk',
    'DoS GoldenEye', 'Heartbleed'
]
