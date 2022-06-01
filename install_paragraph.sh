cd ..
for prj in paragraph-core hlo-bridge; do
  git clone git@github.com:paragraph-sim/${prj} ${prj}
  cd ${prj}
  bazel test -c opt ...
  cd ..
done
for prj in paragraph-creator hlo-examples; do
  git clone git@github.com:paragraph-sim/${prj} ${prj}
done
