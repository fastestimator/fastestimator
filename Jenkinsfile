pipeline {
  agent any
  
  stages {
    
    stage('Build') {
      steps {
        sh './test/install_dependencies.sh'
      }
    }

    stage('Sonarqube/Test') {
      environment {
        scannerHome = tool 'SonarScannerFE'
      }

      steps {
        sh '''
            . /var/lib/jenkins/workspace/venv/bin/activate
            python3 -m pytest --cov-config=/var/lib/jenkins/workspace/fe_git_fastestimator_PR-49/.coveragerc --cov --cov-report xml:coverage.xml ./fastestimator ./fastestimator 
        '''
        withSonarQubeEnv('SonarFE') {
          sh "${scannerHome}/bin/sonar-scanner"
        }
        timeout(time: 10, unit: 'MINUTES') {
          waitForQualityGate abortPipeline: true
        }
      }
    }

  }
}
