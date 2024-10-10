FROM graphpatch-dev
WORKDIR /graphpatch

ENV PYENV_ROOT=/root/.pyenv
ENV PATH=$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN echo "PYENV_ROOT=$PYENV_ROOT" >> /etc/environment \
  && echo "PATH=$PATH" >> /etc/environment

RUN wget -O - https://pyenv.run | bash
RUN pyenv install -s

ENTRYPOINT [ "/init/tailscale_init.sh" ]
