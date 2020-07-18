from flask import Flask, request, make_response, send_file, send_from_directory, jsonify, abort
import json, uuid, re, pandas as pd, numpy as np

import sqlite3
from sqlite3 import Error
 
def create_connection(path):
    connection = None
    try:
        connection = sqlite3.connect(path)
        print("Connection to SQLite DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection

conn = create_connection('aitesting.db')

def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query executed successfully: " + query)
    except Error as e:
        print(f"The error '{e}' occurred")

    return cursor.lastrowid

def execute_read_query(connection, query):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except:
        return("The error occurred")

def set_ai_value(data, parent=None, child=None):
    if parent is None:
        try:
            return data[child]
        except Exception:
            return 'N/A'
    else:
        try:
            return data[parent][child]
        except Exception:
            return 'N/A'       

app = Flask(__name__)

###
# @app.route('/auth', methods=['GET'])
# def get_study():
# get_tokens = "SELECT * FROM services;"
# return json.dumps({'tokens': execute_read_query(conn, get_tokens) })
@app.route('/', methods=['GET'])
def welcome():
    return jsonify(message='Welcome to AI testing platform. For details send request to Nikolay Pavlov, n.pavlov@npcmr.ru.'), 200

#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
@app.route('/auth', methods=['POST'])
def set_session_token():
    try:
        data = request.json
        service_name = data['name']
        token = data['token']
        dataset = data['dataset']
    except Exception:
        abort(400, 'POST request content is incorrect (should contain name, token, dataset)')

    service = execute_read_query(conn, f"SELECT id,token FROM services WHERE name='{service_name}' LIMIT 1;")

    # check for empty response
    if (len(service) == 0):
        abort(401, 'Incorrect name or token')

    # take only first row
    service = service[0]
    
    # insert new session_token:
    try:
        if service[1] == token:
            # first make all previous session inactive:
            execute_query(conn, f"UPDATE session_tokens SET active=FALSE WHERE service='{service[0]}';")
            
            # generate session_token
            session_token = uuid.uuid4()

            # put session_token to db
            create_users = f"""
                INSERT INTO session_tokens 
                    (service, session_token, issue_date, expiry_date, active) 
                VALUES 
                    ('{service[0]}', '{session_token}', datetime('now'), datetime('now', '1 days'), TRUE);
                """
            session = execute_query(conn, create_users)
        else:
            abort()
    except:
        abort(401, 'Incorrect name or token')


    # get list of ids for selected datasets
    ds = pd.read_sql_query(f"SELECT id, title FROM datasets WHERE title LIKE '{dataset}'", conn)

    # check for empty response
    if len(ds.index) == 0:
        abort(404, 'Dataset not found')

    # shuffle list
    ids = ds.loc[ds.title==dataset].id.astype('int').sample(frac=1).to_list()

    # add testing items to testing table
    for f in ids:
        create_list = f"""
                INSERT INTO testing 
                    (session, dataset_title, dataset_file_id, created, requests) 
                VALUES 
                    ('{session}', '{dataset}', '{f}', datetime('now'), 0);
                """
        execute_query(conn, create_list)

    return jsonify(
        session_token=str(session_token),
        expire_in='24 hours', 
        number_of_items=len(ids)
    ), 200

#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
@app.route('/pull', methods=['POST'])
def pull_study():
    try:
        data = request.json
        service_name = data['name']
        session_token = data['session_token']
    except Exception:
        abort(400, 'POST request content is incorrect (should contain name, session_token).')

    # check is_active and not is_expired
    session = check_token(conn, service_name, session_token)

    # retrieve first empty row inside session
    ds = pd.read_sql_query(f"""
        SELECT 
            t.id AS id,
            t.requests AS requests,
            t.dataset_title AS dataset_title,
            d.filename AS filename
        FROM testing AS t
        LEFT JOIN datasets AS d
            ON t.dataset_file_id=d.id
        WHERE t.received IS NULL 
            AND t.session = {session} 
        ORDER BY t.id LIMIT 1
    
    """, conn)

    # check for empty response
    if len(ds.index) == 0:
        abort(404, 'No available items for this session_token')
    else:
        ds = ds.iloc[0]
    

    # update time of retrieval
    update_time_of_retrieval = f"""
        UPDATE testing 
        SET
            retrieved=datetime('now'),
            requests={ds.requests+1}
        WHERE id={ds.id};
        """
    execute_query(conn, update_time_of_retrieval)

    resp = make_response(send_file(f'datasets/{ds.dataset_title}/{ds.filename}', 'application/zip'))
    resp.headers['testing_item_id'] = re.match(r'^(.*)\.zip$', ds.filename)[1]
    return(resp)

#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
@app.route('/push', methods=['POST'])
def push_study():
    try:
        data = request.json
        service_name = data['name']
        session_token = data['session_token']
        testing_item_id = data['testing_item_id']
        ai_ct = data['ct']
        ai_left_affected_part = set_ai_value(data, parent='left', child='affected_part')
        ai_left_total_volume = set_ai_value(data, parent='left', child='total_volume')
        ai_left_affected_volume = set_ai_value(data, parent='left', child='affected_volume')
        ai_right_affected_part = set_ai_value(data, parent='right', child='affected_part')
        ai_right_total_volume = set_ai_value(data, parent='right', child='total_volume')
        ai_right_affected_volume = set_ai_value(data, parent='right', child='affected_volume')
        viewer_url = set_ai_value(data, child='viewer_url')
        description = set_ai_value(data, child='description')
    except Exception as e:
        abort(400, 'POST request content is incorrect (should contain service_name, session_token, testing_item_id, ai_ct, [ai_left_affected_part], [ai_left_total_volume], [ai_left_affected_volume], [ai_right_affected_part], [ai_right_total_volume], [ai_right_affected_volume], [viewer_url], [description]}).'+str(e))

  

    # check is_active and not is_expired
    session = check_token(conn, service_name, session_token)

    # retrieve all available rows
    ds = pd.read_sql_query(f"""
        SELECT 
            COUNT(*) as testing_items_count
        FROM testing
        WHERE received IS NULL 
            AND session = {session}     
    """, conn)

    # check for empty response
    if len(ds.index) == 0:
        abort(404, 'No available items for this session_token')
    else:
        testing_items_count = str(ds.iloc[0]['testing_items_count'])

    # retrieve right row inside session
    ds = pd.read_sql_query(f"""
        SELECT 
            t.id AS id
        FROM testing AS t
        LEFT JOIN datasets AS d
            ON t.dataset_file_id=d.id
        WHERE t.received IS NULL
            AND d.filename = '{testing_item_id + '.zip'}'
            AND t.session = {session} 
        ORDER BY t.id LIMIT 1
    
    """, conn)

    # check for empty response
    if len(ds.index) == 0:
        abort(404, 'Testing item not found; testing items still available: ' + testing_items_count)
    else:
        ds = ds.iloc[0]

    # update time of retrieval
    update_response = f"""
        UPDATE testing 
        SET
            received=datetime('now'),
            ai_ct = '{ai_ct}',
            ai_left_affected_part = '{ai_left_affected_part}',
            ai_left_total_volume = '{ai_left_total_volume}',
            ai_left_affected_volume = '{ai_left_affected_volume}',
            ai_right_affected_part = '{ai_right_affected_part}',
            ai_right_total_volume = '{ai_right_total_volume}',
            ai_right_affected_volume = '{ai_right_affected_volume}',
            viewer_url = '{viewer_url}',
            description = '{description}'
        WHERE id={ds.id};
        """
    updated_item = execute_query(conn, update_response)

    # retrieve all available rows
    ds = pd.read_sql_query(f"""
        SELECT 
            CAST ((julianday(received)-julianday(retrieved)) * 24 * 60 * 60 AS INTEGER) AS diff_time
        FROM testing
        WHERE id={ds.id}     
    """, conn)

    # check for empty response
    if len(ds.index) == 0:
        abort(404, 'Testing item not found; testing items still available: ' + testing_items_count)
    else:
        diff_time = str(ds.iloc[0]['diff_time'])

    return jsonify(
        message='Results for testing_item_id=' + testing_item_id + ' have been accepted',
        time_to_response=f'{diff_time}s'
    ), 200

def check_token(conn, service_name, session_token):
    session_info = f"""
        SELECT 
            CASE WHEN 
                CAST(julianday('now')-julianday(st.expiry_date) AS TYPE FLOAT) < 0 
                THEN 0 
                ELSE 1 
            END AS is_expired,
            st.id AS session_id,
            st.active AS is_active
        FROM session_tokens AS st 
        LEFT JOIN services 
            AS s ON s.id=st.service 
        WHERE s.name LIKE '{service_name}' 
            AND st.session_token LIKE '{session_token}';
        """
    ds = pd.read_sql_query(session_info, conn)
    
    # check for empty response
    if len(ds.index) == 0:
        abort(401, 'Session token is invalid')
    else:
        ds = ds.iloc[0]

    if np.int32(ds.is_active) != 1:
        abort(401, 'Session token is expired (new session token obtained)')

    try:
        is_expired = np.int32(ds.is_expired)
        if (is_expired):
            abort(401, 'Session token is expired (by expiration date)')
    except:
        abort(400, 'DB returned bad request')

    return np.int32(ds.session_id)

@app.errorhandler(400)
def api_400(error_message):
	return jsonify(
		error_message=str(error_message)
	), 400

@app.errorhandler(401)
def api_401(error_message):
	return jsonify(
		error_message=str(error_message)
	), 401

@app.errorhandler(404)
def api_404(error_message):
	return jsonify(
		error_message=str(error_message)
	), 404

@app.errorhandler(405)
def api_405(error_message):
	return jsonify(
		error_message=str(error_message)
	), 405

@app.errorhandler(500)
def api_500(error_message):
	return jsonify(
		error_message=str(error_message)
	), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
