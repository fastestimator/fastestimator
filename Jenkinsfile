pipeline {
  agent any
  
  stages {
    
    stage('Build') {
      steps {
        sh './test/install_dependencies.sh'
        sh 'python3 -m pytest --cov --cov-report xml:coverage.xml ./ ./'
      }
    }

    stage('Sonarqube') {
      environment {
        scannerHome = tool 'SonarScannerFE'
      }

      steps {
        withSonarQubeEnv('SonarFE') {
          sh "${scannerHome}/bin/sonar-scanner"
        }
        timeout(time: 10, unit: 'MINUTES') {
          waitForQualityGate abortPipeline: true
        }
      }
    }

    stage('Test') {
        steps {
            sh '''
                . /var/lib/jenkins/workspace/venv/bin/activate
                python3 -m pytest
            '''
        }
    }

  }
}
