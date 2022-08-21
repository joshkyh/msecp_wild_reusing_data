import os, json, pyodbc
import time

def get_project_path ():
    PROJECT_PATH = os.path.abspath('.')
    REPO_NAME = 'multi-td'
    PROJECT_PATH = PROJECT_PATH[0:PROJECT_PATH.find(REPO_NAME) + len(REPO_NAME)]
    return(PROJECT_PATH)
    
def readSecret():
    '''
        Reads config from ~/data/secrets/secret.cfg and returns it.
        Config file must look like {"username":"test", "password": "pass", ... }.
        Config file must have following keys: username, password, database, server, driver
    '''

    configDirectory = os.path.dirname(os.path.realpath(__file__))

    with open(configDirectory + '/../data/secrets/secret.cfg') as f:
        return json.loads(f.read())



def get_connection():
# Creating a connection
    configDic = readSecret()
    
    driver = configDic['driver']
    
    server = configDic['server']
    username = configDic['username']
    password = configDic['password']
    database = configDic['database']
    
    while True:
        try:
            # do stuff
            cnxn = pyodbc.connect(f'''DRIVER={driver};
                    SERVER={server};
                    UID={username};
                    PWD={password};
                    DATABASE={database};''')
    
        except:
            print('pyodbc error, try again.')	
            time.sleep(60)
            continue
        break
    

    return cnxn


if __name__ == '__main__':
    cnxn = get_connection()