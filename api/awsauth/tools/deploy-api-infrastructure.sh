

execute() {
  cat > .env << EOF
${{ secrets.DOTENV }}
EOF

  sls config credentials \
      --provider aws \
      --key $AWS_ACCESS_KEY_ID \
      --secret AWS_SECRET_ACCESS_KEY

  # Install python requirements and pare python libraries to fit Cloudfront lambda
  # 1MB max package size
  sls requirements install
  find .requirements | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
  rm .requirements/**/*.so
  sls deploy -s $ENV
}
