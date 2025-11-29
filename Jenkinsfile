pipeline {
    agent any

    environment {
        // These will be stored in Jenkins Credentials
        AWS_ACCESS_KEY_ID     = credentials('AWS_ACCESS_KEY_ID')
        AWS_SECRET_ACCESS_KEY = credentials('AWS_SECRET_ACCESS_KEY')
    }

    stages {

        stage('Checkout Code') {
            steps {
                checkout scm
            }
        }

        stage('Set up Python') {
            steps {
                sh '''
                sudo apt-get update
                sudo apt-get install -y python3 python3-pip python3-venv
                python3 -m venv venv
                source venv/bin/activate
                pip install --upgrade pip
                '''
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                source venv/bin/activate
                pip install -r requirements.txt
                pip install "dvc[s3]"
                '''
            }
        }

        stage('DVC Pull from S3') {
            steps {
                sh '''
                source venv/bin/activate
                dvc pull -r s3_remote
                '''
            }
        }

        stage('Run MLflow Pipeline') {
            steps {
                sh '''
                source venv/bin/activate
                python src/mlflow_pipeline.py
                '''
            }
        }

    }

    post {
        always {
            echo "Archiving artifacts..."

            archiveArtifacts artifacts: 'models/**', fingerprint: true
            archiveArtifacts artifacts: 'metrics/**', fingerprint: true
            archiveArtifacts artifacts: 'data/**', fingerprint: true
        }
    }
}
